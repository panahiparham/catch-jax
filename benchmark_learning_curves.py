"""Benchmark: DQN vs. random agent on Catch and Dancing Catch.

Run with::

    uv run --group benchmark python benchmark_learning_curves.py
"""

from __future__ import annotations

import time
from functools import partial

import numpy as np

import benchmark_common as bm
from catch_jax import Catch, CatchParams, DancingCatch, DancingCatchParams
from catch_jax.constants import NUM_ACTIONS

N_SEEDS = 30
CATCH_TOTAL_TIMESTEPS = 50_000
DANCING_CATCH_TOTAL_TIMESTEPS = 500_000
SWAP_EVERY = 10_000


def run_panel(name, env, env_params, obs_dim, total_timesteps):
    """Run DQN and random agents on one environment; returns a plot panel dict."""
    random_train = partial(bm.random_train, env=env, env_params=env_params,
                           obs_dim=obs_dim, action_dim=NUM_ACTIONS,
                           total_timesteps=total_timesteps)
    dqn_train = partial(bm.dqn_train, env=env, env_params=env_params,
                        obs_dim=obs_dim, action_dim=NUM_ACTIONS,
                        total_timesteps=total_timesteps)

    results = {}
    for agent_name, fn in [("DQN", dqn_train), ("Random", random_train)]:
        t = time.perf_counter()
        print(f"Running {name} {agent_name}...")
        results[agent_name] = bm.run(fn, N_SEEDS)
        elapsed = time.perf_counter() - t
        print(f"  {agent_name}: {N_SEEDS} seeds x {total_timesteps} steps in {elapsed:.1f}s")

    return {
        "name": name,
        "dqn_rewards": results["DQN"],
        "random_rewards": results["Random"],
        "total_timesteps": total_timesteps,
        "n_seeds": N_SEEDS,
    }


def main():
    t_start = time.perf_counter()

    catch_env = Catch()
    catch_params = CatchParams()
    catch_obs_dim = int(np.prod(catch_env.observation_space(catch_params).shape))
    catch_panel = run_panel("Catch", catch_env, catch_params, catch_obs_dim, CATCH_TOTAL_TIMESTEPS)

    dancing_env = DancingCatch()
    dancing_params = DancingCatchParams(swap_every=SWAP_EVERY)
    dancing_obs_dim = int(np.prod(dancing_env.observation_space(dancing_params).shape))
    dancing_panel = run_panel("Dancing Catch", dancing_env, dancing_params, dancing_obs_dim,
                               DANCING_CATCH_TOTAL_TIMESTEPS)

    elapsed_total = time.perf_counter() - t_start
    print(f"\nTotal time: {elapsed_total:.1f}s")

    bm.make_learning_curves_plot([catch_panel, dancing_panel], "benchmark_learning_curves.pdf")


if __name__ == "__main__":
    main()
