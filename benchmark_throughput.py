"""Benchmark environment and agent throughput: numpy vs. JAX, single vs. vmapped.

Measures steps/second for:
- Numpy Catch with random actions in a plain Python loop
- JAX Catch with random actions in jitted scan, single and vmapped
- DQN and random agents on JAX Catch

Run with::

    uv run --group benchmark python benchmark_throughput.py
"""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import benchmark_common as bm
from catch_jax import Catch, CatchParams
from catch_jax.catch import NUM_ACTIONS

# The numpy baseline is the vendored csuite implementation that also serves as
# the test oracle, so there is one copy of it in the repo.
sys.path.insert(0, str(Path(__file__).parent / "tests"))
from _reference_catch import Catch as NumpyCatch

env_jax = Catch()
env_params = CatchParams()
OBS_SPACE = env_jax.observation_space(env_params)
OBS_DIM = int(np.prod(OBS_SPACE.shape))
ACTION_DIM = NUM_ACTIONS


def format_throughput(steps, elapsed_s):
    """Format throughput with thousands separators and appropriate precision."""
    throughput = steps / elapsed_s
    if throughput >= 100_000:
        return f"{int(round(throughput / 1000)) * 1000:,}"
    elif throughput >= 10_000:
        return f"{int(round(throughput / 100)) * 100:,}"
    else:
        return f"{int(round(throughput)):,}"


def format_speedup(speedup):
    """Format speedup with appropriate precision."""
    if speedup >= 10:
        return f"{int(round(speedup))}x"
    else:
        return f"{speedup:.1f}x"


def benchmark_numpy(num_steps):
    """Benchmark numpy reference in a plain Python loop."""
    env = NumpyCatch(spawn_probability=0.1, seed=0)
    env.reset()
    rng = np.random.default_rng(0)

    for _ in range(num_steps):
        env.step(rng.integers(0, 3))

    return num_steps


def benchmark_jax_single(num_steps):
    """Benchmark JAX Catch, single env, in jitted scan with random policy.

    Excludes compilation time via warmup, includes blocking time.
    """
    train_fn_single = partial(
        bm.random_train,
        env=env_jax,
        env_params=env_params,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        total_timesteps=num_steps
    )

    rng = jax.random.key(0)
    jitted_fn = jax.jit(train_fn_single)

    jax.block_until_ready(jitted_fn(rng))

    t_start = time.perf_counter()
    jax.block_until_ready(jitted_fn(rng))
    elapsed = time.perf_counter() - t_start

    return num_steps, elapsed


def benchmark_jax_vmapped(num_envs, num_steps):
    """Benchmark JAX Catch, vmapped over num_envs envs, in jitted scan.

    Excludes compilation time via warmup, includes blocking time.
    """
    train_fn_single = partial(
        bm.random_train,
        env=env_jax,
        env_params=env_params,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        total_timesteps=num_steps
    )

    vmapped_train = jax.vmap(train_fn_single)
    jitted_vmapped = jax.jit(vmapped_train)

    keys = jax.random.split(jax.random.key(0), num_envs)

    jax.block_until_ready(jitted_vmapped(keys))

    t_start = time.perf_counter()
    jax.block_until_ready(jitted_vmapped(keys))
    elapsed = time.perf_counter() - t_start

    return num_envs * num_steps, elapsed


def benchmark_agent(num_steps, n_seeds, agent_name):
    """Benchmark an agent (random or DQN) for throughput.

    Excludes compilation time via warmup, includes blocking time.
    """
    if agent_name == "Random":
        train_fn = partial(bm.random_train,
                          env=env_jax, env_params=env_params,
                          obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                          total_timesteps=num_steps)
    else:
        train_fn = partial(bm.dqn_train,
                          env=env_jax, env_params=env_params,
                          obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                          total_timesteps=num_steps)

    keys = jax.vmap(jax.random.key)(jnp.arange(n_seeds))
    vmapped_fn = jax.vmap(train_fn)
    jitted_fn = jax.jit(vmapped_fn)

    jax.block_until_ready(jitted_fn(keys))

    t_start = time.perf_counter()
    jax.block_until_ready(jitted_fn(keys))
    elapsed = time.perf_counter() - t_start

    return n_seeds * num_steps, elapsed


def main():
    print("\n" + "=" * 80)
    print("THROUGHPUT BENCHMARKS: numpy vs. JAX Catch")
    print("=" * 80)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    numpy_steps = 10_000
    print(f"Benchmarking numpy Catch with {numpy_steps:,} steps...")
    t_start = time.perf_counter()
    benchmark_numpy(numpy_steps)
    numpy_elapsed = time.perf_counter() - t_start
    numpy_throughput = numpy_steps / numpy_elapsed
    print(f"  Numpy: {format_throughput(numpy_steps, numpy_elapsed)} steps/sec")
    print()

    print("Table 1: Environment Throughput (Random Policy)")
    print("-" * 80)

    rows = []
    rows.append({
        "impl": "csuite (numpy)",
        "n_envs": 1,
        "throughput": format_throughput(numpy_steps, numpy_elapsed),
        "speedup": "1x"
    })

    jax_single_steps = 100_000
    print(f"Benchmarking JAX single env with {jax_single_steps:,} steps...")
    jax_single_total, jax_single_elapsed = benchmark_jax_single(jax_single_steps)
    jax_single_throughput = jax_single_total / jax_single_elapsed
    jax_single_speedup = jax_single_throughput / numpy_throughput
    jax_single_display = format_throughput(jax_single_total, jax_single_elapsed)
    jax_single_speedup_display = format_speedup(jax_single_speedup)
    print(f"  JAX 1x1: {jax_single_display} steps/sec ({jax_single_speedup_display})")
    rows.append({
        "impl": "catch-jax",
        "n_envs": 1,
        "throughput": format_throughput(jax_single_total, jax_single_elapsed),
        "speedup": format_speedup(jax_single_speedup)
    })

    vmapped_configs = [
        (8, 100_000),
        (64, 100_000),
        (512, 100_000),
        (4096, 100_000)
    ]

    for n_envs, steps in vmapped_configs:
        print(f"Benchmarking JAX vmapped {n_envs} envs with {steps:,} steps per env...")
        total_steps, elapsed = benchmark_jax_vmapped(n_envs, steps)
        throughput = total_steps / elapsed
        speedup = throughput / numpy_throughput
        throughput_display = format_throughput(total_steps, elapsed)
        speedup_display = format_speedup(speedup)
        print(f"  JAX {n_envs}x: {throughput_display} steps/sec ({speedup_display})")
        rows.append({
            "impl": "catch-jax",
            "n_envs": n_envs,
            "throughput": format_throughput(total_steps, elapsed),
            "speedup": format_speedup(speedup)
        })

    print()
    print("| Implementation | Environments | Steps/sec | Speedup vs. numpy |")
    print("|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['impl']} | {row['n_envs']} | {row['throughput']} | "
            f"{row['speedup']} |"
        )

    print()
    print("Table 2: Agent Throughput on catch-jax")
    print("-" * 80)

    agent_results = {}

    agent_steps = 50_000
    for name in ["Random", "DQN"]:
        print(f"Benchmarking {name} agent with {agent_steps:,} steps...")
        total_steps, elapsed = benchmark_agent(agent_steps, 1, name)
        print(f"  {name}: {format_throughput(total_steps, elapsed)} steps/sec")
        agent_results[name] = (total_steps, elapsed)

    print()
    print("| Agent | Steps/sec |")
    print("|---|---|")
    for name in ["Random", "DQN"]:
        total_steps, elapsed = agent_results[name]
        print(f"| {name} | {format_throughput(total_steps, elapsed)} |")

    print()
    print("=" * 80)
    print("Benchmarking complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
