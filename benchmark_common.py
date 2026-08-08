"""Shared benchmark code for DQN vs. random agents."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.lax
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import optax

# --- DQN hyperparameters (identical for both Catch and Dancing Catch) --------

LR = 0.01
BUFFER_SIZE = 100_000
BATCH_SIZE = 32
UPDATE_FREQ = 4
TARGET_REFRESH = 128
GAMMA = 0.9
EPSILON = 0.01
HIDDEN_SIZE = 32
EMA_BETA = 0.99
PLOT_POINTS = 500

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
    terminated: jax.Array
    pos: jax.Array
    size: jax.Array


def buffer_init(obs_dim):
    """Initialize replay buffer."""
    z = jnp.zeros((BUFFER_SIZE, obs_dim), dtype=jnp.float32)
    return Buffer(
        obs=z, action=jnp.zeros(BUFFER_SIZE, jnp.int32),
        reward=jnp.zeros(BUFFER_SIZE), next_obs=z,
        terminated=jnp.zeros(BUFFER_SIZE), pos=jnp.int32(0), size=jnp.int32(0),
    )


def buffer_add(b, obs, action, reward, next_obs, terminated):
    """Add one transition to the buffer."""
    i = b.pos % BUFFER_SIZE
    return Buffer(
        obs=b.obs.at[i].set(obs), action=b.action.at[i].set(action),
        reward=b.reward.at[i].set(reward), next_obs=b.next_obs.at[i].set(next_obs),
        terminated=b.terminated.at[i].set(terminated),
        pos=b.pos + 1, size=jnp.minimum(b.size + 1, BUFFER_SIZE),
    )


def buffer_sample(b, key):
    """Sample a batch from the buffer."""
    idx = jax.random.randint(key, (BATCH_SIZE,), 0, b.size)
    return b.obs[idx], b.action[idx], b.reward[idx], b.next_obs[idx], b.terminated[idx]


# --- metrics carry structure -------------------------------------------------

class MetricsCarry(NamedTuple):
    ema_reward: jax.Array
    ema_catchrate: jax.Array
    n_steps: jax.Array
    n_resolved: jax.Array


# --- one agent-environment interaction, per seed ----------------------------

def random_train(rng, env, env_params, obs_dim, action_dim, total_timesteps):
    """Run random agent: uniform action sampling, no learning."""
    rng, k = jax.random.split(rng)
    obs, state = env.reset(k, env_params)
    obs = obs.reshape(obs_dim)

    def step(carry, _):
        state, obs, rng = carry
        rng, k_a, k_step = jax.random.split(rng, 3)
        action = jax.random.randint(k_a, (), 0, action_dim, dtype=jnp.int32)
        next_obs, next_state, reward, _, _, _ = env.step(k_step, state, action, env_params)
        next_obs = next_obs.reshape(obs_dim)
        return (next_state, next_obs, rng), reward

    _, rewards = jax.lax.scan(step, (state, obs, rng), jnp.arange(total_timesteps))
    return rewards


def dqn_train(rng, env, env_params, obs_dim, action_dim, total_timesteps):
    """Run DQN agent with epsilon-greedy exploration and Q-learning."""
    rng, k_init, k_reset = jax.random.split(rng, 3)
    params = init_mlp(k_init, [obs_dim, HIDDEN_SIZE, HIDDEN_SIZE, action_dim])
    target = params
    opt_state = optimizer.init(params)
    buffer = buffer_init(obs_dim)
    obs, state = env.reset(k_reset, env_params)
    obs = obs.reshape(obs_dim)

    def step(carry, t):
        params, target, opt_state, buffer, state, obs, rng = carry
        rng, k_a, k_expl, k_step, k_sample = jax.random.split(rng, 5)

        greedy = jnp.argmax(mlp(params, obs)).astype(jnp.int32)
        rand_a = jax.random.randint(k_a, (), 0, action_dim, dtype=jnp.int32)
        action = jnp.where(jax.random.uniform(k_expl) < EPSILON, rand_a, greedy)

        next_obs, next_state, reward, terminated, _, _ = env.step(k_step, state, action, env_params)
        next_obs = next_obs.reshape(obs_dim)
        buffer = buffer_add(buffer, obs, action, reward, next_obs, terminated.astype(jnp.float32))

        def do_train(params, opt_state):
            b_obs, b_a, b_r, b_nobs, b_term = buffer_sample(buffer, k_sample)

            def loss_fn(p):
                q_a = jnp.take_along_axis(mlp(p, b_obs), b_a[:, None], axis=-1).squeeze(-1)
                bootstrap = (1.0 - b_term) * jnp.max(mlp(target, b_nobs), axis=-1)
                target_q = b_r + GAMMA * bootstrap
                return jnp.mean((q_a - jax.lax.stop_gradient(target_q)) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            return optax.apply_updates(params, updates), opt_state

        can_train = (t >= UPDATE_FREQ - 1) & (t % UPDATE_FREQ == UPDATE_FREQ - 1) & (buffer.size >= BATCH_SIZE)
        params, opt_state = jax.lax.cond(
            can_train,
            lambda: do_train(params, opt_state),
            lambda: (params, opt_state),
        )

        target = jax.lax.cond(
            t % TARGET_REFRESH == TARGET_REFRESH - 1,
            lambda: params,
            lambda: target,
        )

        carry = (params, target, opt_state, buffer, next_state, next_obs, rng)
        return carry, reward

    carry0 = (params, target, opt_state, buffer, state, obs, rng)
    _, rewards = jax.lax.scan(step, carry0, jnp.arange(total_timesteps))
    return rewards


def _compute_emas(rewards):
    """Bias-corrected reward EMA and catch-rate EMA from one seed's reward sequence.

    :param rewards: [T] array of rewards (-1, 0, or +1)
    :return: ([T], [T]) pair of bias-corrected reward EMA and catch-rate EMA
    """
    def step(carry, reward):
        ema_r, ema_c, n_steps, n_resolved = carry

        ema_r = EMA_BETA * ema_r + (1.0 - EMA_BETA) * reward
        n_steps = n_steps + 1

        resolved = reward != 0.0
        catchrate_label = (reward > 0.0).astype(jnp.float32)
        ema_c = jnp.where(
            resolved,
            EMA_BETA * ema_c + (1.0 - EMA_BETA) * catchrate_label,
            ema_c
        )
        n_resolved = n_resolved + resolved.astype(jnp.int32)

        ema_r_corrected = ema_r / (1.0 - EMA_BETA**jnp.maximum(n_steps, 1))
        ema_c_corrected = ema_c / (1.0 - EMA_BETA**jnp.maximum(n_resolved, 1))

        return (ema_r, ema_c, n_steps, n_resolved), (ema_r_corrected, ema_c_corrected)

    carry0 = (jnp.float32(0.0), jnp.float32(0.0), jnp.int32(0), jnp.int32(0))
    _, (ema_rs, ema_cs) = jax.lax.scan(step, carry0, rewards)
    return ema_rs, ema_cs


_compute_emas_batched = jax.jit(jax.vmap(_compute_emas))


def compute_emas(rewards):
    """Bias-corrected reward EMA and catch-rate EMA for a batch of seeds.

    :param rewards: [N_SEEDS, T] array of rewards (-1, 0, or +1)
    :return: ([N_SEEDS, T], [N_SEEDS, T]) pair of bias-corrected EMAs
    """
    ema_rs, ema_cs = _compute_emas_batched(jnp.asarray(rewards))
    return np.asarray(ema_rs), np.asarray(ema_cs)


def run(train_fn, n_seeds):
    """Run one agent for n_seeds seeds; returns [n_seeds, T] reward array."""
    keys = jax.vmap(jax.random.key)(jnp.arange(n_seeds))
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
        boot_indices = rng.integers(0, n, size=(e - s, n))
        boot[s:e] = stack[boot_indices].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot, [lo, hi], axis=0)
    return mean, ci_lo, ci_hi


def make_plot(dqn_rewards, random_rewards, title, path, total_timesteps, n_seeds, swap_steps=None):
    """Plot reward EMA and catch-rate EMA over time with bootstrap CIs.

    :param dqn_rewards: [N_SEEDS, T] array of DQN rewards
    :param random_rewards: [N_SEEDS, T] array of random agent rewards
    :param title: plot title
    :param path: output file path
    :param total_timesteps: total timesteps in the run
    :param n_seeds: number of seeds
    :param swap_steps: optional list of steps to mark with vertical guidelines
    """
    if swap_steps is None:
        swap_steps = []

    dqn_ema_rs, dqn_ema_cs = compute_emas(dqn_rewards)
    random_ema_rs, random_ema_cs = compute_emas(random_rewards)

    idx = np.linspace(0, total_timesteps - 1, PLOT_POINTS).astype(int)
    dqn_ema_rs, dqn_ema_cs = dqn_ema_rs[:, idx], dqn_ema_cs[:, idx]
    random_ema_rs, random_ema_cs = random_ema_rs[:, idx], random_ema_cs[:, idx]

    dqn_r_mean, dqn_r_lo, dqn_r_hi = bootstrap_mean_ci(dqn_ema_rs)
    dqn_c_mean, dqn_c_lo, dqn_c_hi = bootstrap_mean_ci(dqn_ema_cs)
    random_r_mean, random_r_lo, random_r_hi = bootstrap_mean_ci(random_ema_rs)
    random_c_mean, random_c_lo, random_c_hi = bootstrap_mean_ci(random_ema_cs)

    x = idx

    fig, (ax_r, ax_c) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    ax_r.fill_between(x, dqn_r_lo, dqn_r_hi, color="tab:blue", alpha=0.2)
    ax_r.plot(x, dqn_r_mean, lw=2.5, color="tab:blue", label="DQN")
    ax_r.fill_between(x, random_r_lo, random_r_hi, color="tab:red", alpha=0.2)
    ax_r.plot(x, random_r_mean, lw=2.5, color="tab:red", label="Random")

    for step in swap_steps:
        ax_r.axvline(step, color="gray", alpha=0.15, lw=0.8, zorder=0)

    ax_r.set_ylabel("Reward EMA", rotation=0, ha="right", va="center", labelpad=12)
    ax_r.grid(False)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)

    ax_c.fill_between(x, dqn_c_lo, dqn_c_hi, color="tab:blue", alpha=0.2)
    ax_c.plot(x, dqn_c_mean, lw=2.5, color="tab:blue", label="DQN")
    ax_c.fill_between(x, random_c_lo, random_c_hi, color="tab:red", alpha=0.2)
    ax_c.plot(x, random_c_mean, lw=2.5, color="tab:red", label="Random")

    for step in swap_steps:
        ax_c.axvline(step, color="gray", alpha=0.15, lw=0.8, zorder=0)

    ax_c.set_xlabel("Timestep")
    ax_c.set_ylabel("Catch-rate EMA", rotation=0, ha="right", va="center", labelpad=12)
    ax_c.grid(False)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    if swap_steps:
        swap_proxy = Line2D([], [], color="gray", alpha=0.6, lw=1.2)
        handles, labels = ax_r.get_legend_handles_labels()
        ax_r.legend(handles + [swap_proxy], labels + ["Observation swap"], loc="best", frameon=False)
        ax_c.legend(loc="best", frameon=False)
    else:
        ax_r.legend(loc="best", frameon=False)
        ax_c.legend(loc="best", frameon=False)

    fig.suptitle(title, y=0.995)
    fig.tight_layout()

    for ext in (".png", ".pdf"):
        p = path.replace(".pdf", ext)
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"saved {p}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: Final metrics (mean ± std across {n_seeds} seeds)")
    print("=" * 70)
    print(f"{'Metric':<25} {'DQN':<30} {'Random':<30}")
    print("-" * 70)
    print(f"{'Reward EMA':<25} {dqn_r_mean[-1]:>7.4f} ± {np.std(dqn_ema_rs[:, -1]):>6.4f}  {random_r_mean[-1]:>7.4f} ± {np.std(random_ema_rs[:, -1]):>6.4f}")
    print(f"{'Catch-rate EMA':<25} {dqn_c_mean[-1]:>7.4f} ± {np.std(dqn_ema_cs[:, -1]):>6.4f}  {random_c_mean[-1]:>7.4f} ± {np.std(random_ema_cs[:, -1]):>6.4f}")
    print("=" * 70)
