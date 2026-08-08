"""Benchmark: DQN vs. random agent on Dancing Catch.

Run with::

    uv run --group benchmark python benchmark_dancing_catch.py
"""

from __future__ import annotations

import time
from functools import partial

import numpy as np

import benchmark_common as bm
from catch_jax import DancingCatch, DancingCatchParams
from catch_jax.dancing_catch import NUM_ACTIONS

TOTAL_TIMESTEPS = 500_000
N_SEEDS = 30
SWAP_EVERY = 10_000

env = DancingCatch()
env_params = DancingCatchParams(swap_every=SWAP_EVERY)
OBS_SPACE = env.observation_space(env_params)
OBS_DIM = int(np.prod(OBS_SPACE.shape))
ACTION_DIM = NUM_ACTIONS


def main():
    t_start = time.perf_counter()

    random_train = partial(bm.random_train, env=env, env_params=env_params,
                           obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                           total_timesteps=TOTAL_TIMESTEPS)
    dqn_train = partial(bm.dqn_train, env=env, env_params=env_params,
                        obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                        total_timesteps=TOTAL_TIMESTEPS)

    results = {}
    for name, fn in [("DQN", dqn_train), ("Random", random_train)]:
        t = time.perf_counter()
        print(f"Running {name}...")
        results[name] = bm.run(fn, N_SEEDS)
        elapsed = time.perf_counter() - t
        print(f"  {name}: {N_SEEDS} seeds x {TOTAL_TIMESTEPS} steps in {elapsed:.1f}s")

    elapsed_total = time.perf_counter() - t_start
    print(f"\nTotal time: {elapsed_total:.1f}s")

    title = f"DQN vs. Random Agent on Dancing Catch (mean ± 95% bootstrap CI, n={N_SEEDS} seeds)"
    swap_steps = list(range(SWAP_EVERY, TOTAL_TIMESTEPS + 1, SWAP_EVERY))
    bm.make_plot(results["DQN"], results["Random"], title, "benchmark_dancing_catch.pdf",
                 TOTAL_TIMESTEPS, N_SEEDS, swap_steps=swap_steps)


if __name__ == "__main__":
    main()
