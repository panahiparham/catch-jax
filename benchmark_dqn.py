"""Standalone benchmark: DQN vs. a uniform-random agent on Catch (10x5, spawn_probability=0.1).

This benchmark runs both agents for 30 seeds (each is one ``jax.vmap`` over seeds)
over 50k timesteps, then plots two metrics over time with 95% bootstrap
confidence bands and writes ``benchmark_dqn.pdf`` and ``benchmark_dqn.png``.
No experiment harness, no results database — a single self-contained file.

Both metrics are exponential moving averages (β=0.99) tracked inline in the scan:

1. **Reward EMA**: updated every step with the raw reward (0, +1, or -1).
   Expected: random agent ≈ -0.06 (spawn rate 0.1 × net catch rate 0.2),
   working DQN ≈ +0.1 (spawn rate, near-perfect play).

2. **Catch-rate EMA**: event-gated, updated *only* on steps where a ball was
   resolved (reward != 0). Tracks 1.0 for a catch and 0.0 for a miss.
   Expected: random agent ≈ 0.2 (uniform over 5 columns),
   working DQN ≈ 1.0 (near-perfect play).

   On steps where reward == 0 (no ball resolution), catch-rate EMA is unchanged.
   Both EMAs start at 0 and are bias-corrected per-step using the standard
   correction: ema_corrected = ema / (1 - β**n), where n is the count of updates
   (not timesteps; for catch-rate, n is the count of resolved balls).

Since Catch is a continuing environment (no terminal states), every Q-target
bootstraps normally — no terminal masking of the next-state value.

Hyperparameters from NeverEndingRL's ``position_catch/Catch50k/DQN.json``:
- Total timesteps: 50,000
- Replay buffer: 100,000 capacity, uniform (iid) sampling
- Batch size: 32
- Update frequency: every 4 environment steps
- Target network refresh: hard copy every 128 steps
- Warm-up: updates begin once buffer >= batch size (32 transitions)
- Epsilon: 0.01 (constant epsilon-greedy)
- Optimizer: Adam, lr=0.01, β₁=0.9, β₂=0.999, ε=1e-8
- Network: 2 hidden layers of 32 units each, ReLU activation
- Discount factor: γ=0.9

Run with::

    uv run --group benchmark python benchmark_dqn.py
"""

from __future__ import annotations

import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.lax
import matplotlib.pyplot as plt
import numpy as np
import optax

from catch_jax import Catch, CatchParams
from catch_jax.catch import NUM_ACTIONS

# --- configuration ----------------------------------------------------------

TOTAL_TIMESTEPS = 50_000
N_SEEDS = 30

# DQN hyperparameters
LR = 0.01
BUFFER_SIZE = 100_000
BATCH_SIZE = 32
UPDATE_FREQ = 4              # update every N environment steps
TARGET_REFRESH = 128         # hard target-network copy every N steps
GAMMA = 0.9
EPSILON = 0.01               # constant epsilon-greedy
HIDDEN_SIZE = 32
EMA_BETA = 0.99              # exponential moving average decay
PLOT_POINTS = 500            # points bootstrapped/plotted (matches pinball-jax's GRID density)

env = Catch()
env_params = CatchParams()
OBS_SPACE = env.observation_space(env_params)
OBS_DIM = int(np.prod(OBS_SPACE.shape))
ACTION_DIM = NUM_ACTIONS
optimizer = optax.adam(learning_rate=LR, b1=0.9, b2=0.999, eps=1e-8)


# --- Q-network (a plain MLP as a list of (W, b) params) ---------------------

def init_mlp(key, sizes):
    """Initialize a plain MLP: list of (W, b) tuples with He initialization."""
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        w = jax.random.normal(k, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
        params.append((w, jnp.zeros(fan_out)))
    return params


def mlp(params, x):
    """Q-values for a single obs ``(OBS_DIM,)`` or a batch ``(B, OBS_DIM)``."""
    for w, b in params[:-1]:
        x = jax.nn.relu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


# --- uniform replay buffer (fixed-size, in-JAX) -----------------------------

class Buffer(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    pos: jax.Array      # number of transitions ever added
    size: jax.Array     # number currently stored (<= BUFFER_SIZE)


def buffer_init():
    z = jnp.zeros((BUFFER_SIZE, OBS_DIM), dtype=jnp.float32)
    return Buffer(
        obs=z, action=jnp.zeros(BUFFER_SIZE, jnp.int32),
        reward=jnp.zeros(BUFFER_SIZE), next_obs=z,
        pos=jnp.int32(0), size=jnp.int32(0),
    )


def buffer_add(b, obs, action, reward, next_obs):
    i = b.pos % BUFFER_SIZE
    return Buffer(
        obs=b.obs.at[i].set(obs), action=b.action.at[i].set(action),
        reward=b.reward.at[i].set(reward), next_obs=b.next_obs.at[i].set(next_obs),
        pos=b.pos + 1, size=jnp.minimum(b.size + 1, BUFFER_SIZE),
    )


def buffer_sample(b, key):
    idx = jax.random.randint(key, (BATCH_SIZE,), 0, b.size)
    return b.obs[idx], b.action[idx], b.reward[idx], b.next_obs[idx]


# --- metrics carry structure -------------------------------------------------

class MetricsCarry(NamedTuple):
    ema_reward: jax.Array       # exponential moving average of reward
    ema_catchrate: jax.Array    # exponential moving average of catch rate (event-gated)
    n_steps: jax.Array          # count of steps (for reward EMA bias correction)
    n_resolved: jax.Array       # count of resolved balls (for catch-rate EMA bias correction)


# --- one agent-environment interaction, per seed ----------------------------

def _reset(key):
    obs, state = env.reset(key, env_params)
    obs = obs.reshape(OBS_DIM)
    return obs, state


def _step_env(key, state, action):
    obs, state, reward, term, trunc, info = env.step(key, state, action, env_params)
    obs = obs.reshape(OBS_DIM)
    return obs, state, reward, term, trunc, info


def random_train(rng):
    """Run random agent: uniform action sampling, no learning."""
    rng, k = jax.random.split(rng)
    obs, state = _reset(k)

    def step(carry, _):
        state, obs, rng = carry
        rng, k_a, k_step = jax.random.split(rng, 3)
        action = jax.random.randint(k_a, (), 0, ACTION_DIM, dtype=jnp.int32)
        next_obs, next_state, reward, term, trunc, _ = _step_env(k_step, state, action)
        return (next_state, next_obs, rng), reward

    _, rewards = jax.lax.scan(step, (state, obs, rng), jnp.arange(TOTAL_TIMESTEPS))
    return rewards


def dqn_train(rng):
    """Run DQN agent with epsilon-greedy exploration and Q-learning."""
    rng, k_init, k_reset = jax.random.split(rng, 3)
    params = init_mlp(k_init, [OBS_DIM, HIDDEN_SIZE, HIDDEN_SIZE, ACTION_DIM])
    target = params
    opt_state = optimizer.init(params)
    buffer = buffer_init()
    obs, state = _reset(k_reset)

    def step(carry, t):
        params, target, opt_state, buffer, state, obs, rng = carry
        rng, k_a, k_expl, k_step, k_sample = jax.random.split(rng, 5)

        # Epsilon-greedy action selection
        greedy = jnp.argmax(mlp(params, obs)).astype(jnp.int32)
        rand_a = jax.random.randint(k_a, (), 0, ACTION_DIM, dtype=jnp.int32)
        action = jnp.where(jax.random.uniform(k_expl) < EPSILON, rand_a, greedy)

        # Environment step
        next_obs, next_state, reward, term, trunc, _ = _step_env(k_step, state, action)
        buffer = buffer_add(buffer, obs, action, reward, next_obs)

        # Since Catch is a continuing environment (term is always False),
        # we don't reset on termination; we only reset on truncation if needed.
        # For this benchmark, we run without truncation by using a large
        # max_steps_in_episode, so neither term nor trunc should fire.
        # Still, we keep them in the logic for generality.
        done = term | trunc

        def do_train(params, opt_state):
            b_obs, b_a, b_r, b_nobs = buffer_sample(buffer, k_sample)

            def loss_fn(p):
                q_a = jnp.take_along_axis(mlp(p, b_obs), b_a[:, None], axis=-1).squeeze(-1)
                # No terminal masking since Catch never terminates
                target_q = b_r + GAMMA * jnp.max(mlp(target, b_nobs), axis=-1)
                return jnp.mean((q_a - jax.lax.stop_gradient(target_q)) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            return optax.apply_updates(params, updates), opt_state

        # Update every UPDATE_FREQ steps, once buffer is warm
        can_train = (t >= UPDATE_FREQ - 1) & (t % UPDATE_FREQ == UPDATE_FREQ - 1) & (buffer.size >= BATCH_SIZE)
        params, opt_state = jax.lax.cond(
            can_train,
            lambda: do_train(params, opt_state),
            lambda: (params, opt_state),
        )

        # Hard target refresh
        target = jax.lax.cond(
            t % TARGET_REFRESH == TARGET_REFRESH - 1,
            lambda: params,
            lambda: target,
        )

        carry = (params, target, opt_state, buffer, next_state, next_obs, rng)
        return carry, reward

    carry0 = (params, target, opt_state, buffer, state, obs, rng)
    _, rewards = jax.lax.scan(step, carry0, jnp.arange(TOTAL_TIMESTEPS))
    return rewards


def _compute_emas(rewards):
    """Bias-corrected reward EMA and catch-rate EMA from one seed's reward sequence.

    :param rewards: [T] array of rewards (-1, 0, or +1)
    :return: ([T], [T]) pair of bias-corrected reward EMA and catch-rate EMA
    """
    def step(carry, reward):
        ema_r, ema_c, n_steps, n_resolved = carry

        # Reward EMA: updated every step
        ema_r = EMA_BETA * ema_r + (1.0 - EMA_BETA) * reward
        n_steps = n_steps + 1

        # Catch-rate EMA: updated only when resolved (reward != 0)
        resolved = reward != 0.0
        catchrate_label = (reward > 0.0).astype(jnp.float32)  # 1 for catch, 0 for miss
        ema_c = jnp.where(
            resolved,
            EMA_BETA * ema_c + (1.0 - EMA_BETA) * catchrate_label,
            ema_c
        )
        n_resolved = n_resolved + resolved.astype(jnp.int32)

        # Bias correction
        ema_r_corrected = ema_r / (1.0 - EMA_BETA**jnp.maximum(n_steps, 1))
        ema_c_corrected = ema_c / (1.0 - EMA_BETA**jnp.maximum(n_resolved, 1))

        return (ema_r, ema_c, n_steps, n_resolved), (ema_r_corrected, ema_c_corrected)

    carry0 = (jnp.float32(0.0), jnp.float32(0.0), jnp.int32(0), jnp.int32(0))
    _, (ema_rs, ema_cs) = jax.lax.scan(step, carry0, rewards)
    return ema_rs, ema_cs


# One jitted call, vmapped over seeds, instead of a Python loop calling
# _compute_emas twice per seed (once per return value) as a plain list comp.
_compute_emas_batched = jax.jit(jax.vmap(_compute_emas))


def compute_emas(rewards):
    """Bias-corrected reward EMA and catch-rate EMA for a batch of seeds.

    :param rewards: [N_SEEDS, T] array of rewards (-1, 0, or +1)
    :return: ([N_SEEDS, T], [N_SEEDS, T]) pair of bias-corrected EMAs
    """
    ema_rs, ema_cs = _compute_emas_batched(jnp.asarray(rewards))
    return np.asarray(ema_rs), np.asarray(ema_cs)


def run(train_fn):
    """Run one agent for N_SEEDS seeds; returns [N_SEEDS, T] reward array."""
    keys = jax.vmap(jax.random.key)(jnp.arange(N_SEEDS))
    rewards = jax.jit(jax.vmap(train_fn))(keys)
    return np.asarray(rewards)


def bootstrap_mean_ci(stack, n_boot=10_000, lo=2.5, hi=97.5, seed=0):
    """Mean and percentile-bootstrap CI over seeds.

    :param stack: [N_SEEDS, T] array
    :param n_boot: number of bootstrap samples
    :param lo, hi: percentile bounds for CI
    :param seed: RNG seed for reproducibility
    :return: (mean, ci_lo, ci_hi) each of shape [T]
    """
    n, m = stack.shape
    mean = stack.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, m))
    for s in range(0, n_boot, 1000):
        e = min(s + 1000, n_boot)
        # Sample n seeds with replacement, compute mean for each bootstrap sample
        boot_indices = rng.integers(0, n, size=(e - s, n))
        boot[s:e] = stack[boot_indices].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot, [lo, hi], axis=0)
    return mean, ci_lo, ci_hi


def make_plot(dqn_rewards, random_rewards, path):
    """Plot reward EMA and catch-rate EMA over time with bootstrap CIs."""
    # Compute EMAs for every seed in one jitted, vmapped call each.
    dqn_ema_rs, dqn_ema_cs = compute_emas(dqn_rewards)
    random_ema_rs, random_ema_cs = compute_emas(random_rewards)

    # Downsample to PLOT_POINTS before bootstrapping/plotting. The EMA signal
    # is dense (one value per timestep) and smooth, so subsampling loses no
    # visible detail; skipping this step would feed bootstrap_mean_ci's chunked
    # fancy-indexing arrays of shape (1000, N_SEEDS, TOTAL_TIMESTEPS) — 12 GB at
    # T=50_000 — instead of the ~100 MB this produces. `endpoint=True` guarantees
    # the final index equals TOTAL_TIMESTEPS - 1, so the summary printout below
    # still reports the true final value, not an interpolated one.
    idx = np.linspace(0, TOTAL_TIMESTEPS - 1, PLOT_POINTS).astype(int)
    dqn_ema_rs, dqn_ema_cs = dqn_ema_rs[:, idx], dqn_ema_cs[:, idx]
    random_ema_rs, random_ema_cs = random_ema_rs[:, idx], random_ema_cs[:, idx]

    # Compute bootstrap CIs
    dqn_r_mean, dqn_r_lo, dqn_r_hi = bootstrap_mean_ci(dqn_ema_rs)
    dqn_c_mean, dqn_c_lo, dqn_c_hi = bootstrap_mean_ci(dqn_ema_cs)
    random_r_mean, random_r_lo, random_r_hi = bootstrap_mean_ci(random_ema_rs)
    random_c_mean, random_c_lo, random_c_hi = bootstrap_mean_ci(random_ema_cs)

    x = idx

    fig, (ax_r, ax_c) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Top panel: Reward EMA
    ax_r.fill_between(x, dqn_r_lo, dqn_r_hi, color="tab:blue", alpha=0.2)
    ax_r.plot(x, dqn_r_mean, lw=2.5, color="tab:blue", label="DQN")
    ax_r.fill_between(x, random_r_lo, random_r_hi, color="tab:red", alpha=0.2)
    ax_r.plot(x, random_r_mean, lw=2.5, color="tab:red", label="Random")
    ax_r.set_ylabel("Reward EMA", rotation=0, ha="right", va="center", labelpad=12)
    ax_r.grid(False)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    ax_r.legend(loc="best", frameon=False)

    # Bottom panel: Catch-rate EMA
    ax_c.fill_between(x, dqn_c_lo, dqn_c_hi, color="tab:blue", alpha=0.2)
    ax_c.plot(x, dqn_c_mean, lw=2.5, color="tab:blue", label="DQN")
    ax_c.fill_between(x, random_c_lo, random_c_hi, color="tab:red", alpha=0.2)
    ax_c.plot(x, random_c_mean, lw=2.5, color="tab:red", label="Random")
    ax_c.set_xlabel("Timestep")
    ax_c.set_ylabel("Catch-rate EMA", rotation=0, ha="right", va="center", labelpad=12)
    ax_c.grid(False)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.legend(loc="best", frameon=False)

    title = f"DQN vs. Random Agent on Catch (mean ± 95% bootstrap CI, n={N_SEEDS} seeds)"
    fig.suptitle(title, y=0.995)
    fig.tight_layout()

    # Save PNG and PDF
    for ext in (".png", ".pdf"):
        p = path.replace(".pdf", ext)
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"saved {p}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: Final metrics (mean ± std across 30 seeds)")
    print("=" * 70)
    print(f"{'Metric':<25} {'DQN':<30} {'Random':<30}")
    print("-" * 70)
    print(f"{'Reward EMA':<25} {dqn_r_mean[-1]:>7.4f} ± {np.std(dqn_ema_rs[:, -1]):>6.4f}  {random_r_mean[-1]:>7.4f} ± {np.std(random_ema_rs[:, -1]):>6.4f}")
    print(f"{'Catch-rate EMA':<25} {dqn_c_mean[-1]:>7.4f} ± {np.std(dqn_ema_cs[:, -1]):>6.4f}  {random_c_mean[-1]:>7.4f} ± {np.std(random_ema_cs[:, -1]):>6.4f}")
    print("=" * 70)


def main():
    t_start = time.perf_counter()

    results = {}
    for name, fn in [("DQN", dqn_train), ("Random", random_train)]:
        t = time.perf_counter()
        print(f"Running {name}...")
        results[name] = run(fn)
        elapsed = time.perf_counter() - t
        print(f"  {name}: {N_SEEDS} seeds x {TOTAL_TIMESTEPS} steps in {elapsed:.1f}s")

    elapsed_total = time.perf_counter() - t_start
    print(f"\nTotal time: {elapsed_total:.1f}s")

    make_plot(results["DQN"], results["Random"], "benchmark_dqn.pdf")


if __name__ == "__main__":
    main()
