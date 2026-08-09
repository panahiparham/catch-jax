# catch-jax

This project is a JAX re-implementation of the continuing Catch and its non-stationary
Dancing Catch RL environments [1]. `reset` and `step` are pure functions and are fully JIT- and vmap-able, enabling
distributed training and large-scale experimentation. These environments have recently been used to develop and evaluate continual RL algorithms [2].

## References

[1] [Deepmind's Csuite](https://github.com/google-deepmind/csuite).

[2] Mesbahi, G., Panahi, P. M., Mastikhina, O., Tang, S., White, M., & White, A. (2025).
[Position: Lifetime tuning is incompatible with continual reinforcement learning](https://openreview.net/challenge?redirect=%2Fpdf%3Fid%3DJMoWFkwnvv).
*Forty-second International Conference on Machine Learning (ICML)*.


## Usage

Add it to your project with [uv](https://docs.astral.sh/uv/):

```sh
uv add git+https://github.com/panahiparham/catch-jax
```

```python
import jax
from catch_jax import Catch, CatchParams

env = Catch()  # rows and columns are optional; defaults: 10, 5
params = CatchParams(spawn_probability=0.1, max_steps_in_episode=1000)

key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

# obs: (rows, columns) board, 1.0 for ball/paddle, 0.0 for empty
# actions: 0 = LEFT, 1 = STAY, 2 = RIGHT
obs, state, reward, terminated, truncated, info = env.step(key, state, 1, params)
```

See [`example.py`](example.py) for a jitted `lax.scan` rollout.

`DancingCatch` and `DancingCatchParams` have the same interface, so the same code
works for the variant by swapping the class names.

## Variants

### Catch

**Observation:** A `float32` array of shape `(rows, columns)`, with 1.0 indicating
a ball or paddle and 0.0 indicating an empty cell.

**Actions:** 3 discrete actions: `0 = LEFT`, `1 = STAY`, `2 = RIGHT`. Actions are
applied as `dx = action - 1` with the paddle position clipped to `[0, columns - 1]`.

**Reward:** At each step:
- `+1.0` when a ball reaches the paddle row in the paddle's column
- `-1.0` when a ball reaches the paddle row in any other column
- `0.0` otherwise

**Termination:** The environment is continuing, so `terminated` is always `False`.
`truncated` fires when `timestep >= params.max_steps_in_episode`.

**Dynamics:** The paddle occupies the bottom row. Balls fall one row per step, staying
in the column they spawned in. Each step, with probability `spawn_probability` (default 0.1),
one ball appears at row 0 in a uniformly random column. A ball arriving at the paddle row
is scored and removed in the same step, so it is never drawn there. Within a step the
order is: move the paddle, descend the balls, resolve any ball on the paddle row, then spawn.

**Parameters:** `CatchParams(spawn_probability=0.1, max_steps_in_episode=10**9)`. Board
size is set on the constructor: `Catch(rows=10, columns=5)`.

![Random policy on Catch](catch_random_policy.gif)

The GIF shows a uniform-random policy over 120 steps, with the paddle and balls drawn
in different colours. Regenerate with:

```sh
uv run --group benchmark python make_gif.py
```

### Dancing Catch

Actions, reward, termination, and dynamics are identical to Catch. The observation differs.

**Observation:** A `float32` array of shape `(rows * columns,)`. The board is flattened
and read through a permutation held in `state.shuffle_idx`, so the observation is
`board.reshape(-1)[shuffle_idx]`.

**Non-stationarity:** `reset` starts the permutation at the identity. Every `swap_every`
steps the environment transposes two uniformly drawn entries of `shuffle_idx`, so the
mapping from observation index to board cell drifts over training.

**Parameters:** `DancingCatchParams(spawn_probability=0.1, swap_every=10_000, max_steps_in_episode=10**9)`.

**Rendering:** `render(state)` returns `uint8` of shape `(rows, columns, 3)` built from
the permuted observation, so the image shows the scrambled board.

## Benchmarks

Both benchmarks compare a small DQN agent against a uniform-random baseline across 30 seeds.
The agent implementation and hyperparameters are shared and defined in `benchmark_common.py`.

**Hyperparameters**

The hyperparameters are derived from the NeverEndingRL suite (`position_catch/Catch50k/DQN.json`):

| Hyperparameter | Value |
| --- | --- |
| Total timesteps | Differs per benchmark, given below |
| Seeds | 30 |
| Replay buffer size | 100,000 (uniform, iid sampling) |
| Batch size | 32 |
| Update frequency | Every 4 environment steps |
| Target network refresh | Hard copy every 128 steps |
| Warm-up | Updates begin once buffer >= batch size (32 transitions) |
| Epsilon | 0.01 (constant epsilon-greedy) |
| Optimizer | Adam: learning rate 0.01, beta1=0.9, beta2=0.999, eps=1e-8 |
| Network | 2 hidden layers x 32 units, ReLU activation |
| Discount (gamma) | 0.9 |

**Metrics**

Two exponential moving averages (EMA) are tracked with beta=0.99:

1. **Reward EMA:** Updated every step with the raw reward (-1, 0, or +1). Since Catch is
   continuing (no episode termination), we track the moving average of step rewards to
   assess convergence of the agent's behavior.

2. **Catch-rate EMA:** Event-gated, updated only on steps where a ball is resolved
   (reward != 0). The value is 1.0 for a catch and 0.0 for a miss. On steps where no
   ball resolves (reward = 0), this EMA is unchanged. This metric isolates the agent's
   accuracy when catching is possible.

Both EMAs are bias-corrected at each step using the standard correction:
`ema_corrected = ema / (1 - beta^n)`, where n is the count of updates (different for each EMA).

### Catch

50,000 timesteps, 30 seeds.

![DQN vs. random agent on Catch](benchmark_dqn.png)

Run with:

```sh
uv run --group benchmark python benchmark_dqn.py
```

### Dancing Catch

500,000 timesteps, 30 seeds, `swap_every=10_000` giving 50 observation swaps, which the
plot marks with vertical guidelines.

![DQN vs. random agent on Dancing Catch](benchmark_dancing_catch.png)

Run with:

```sh
uv run --group benchmark python benchmark_dancing_catch.py
```

## Throughput

These measure environment steps per second on CPU. Compilation time is excluded.
The numbers below were measured on an Apple M1 CPU with 8 cores.

Run with:

```sh
uv run --group benchmark python benchmark_throughput.py
```

**Environment throughput, random policy**

The vmapped rows count total steps across all environments.

| Implementation | Environments | Steps/sec | Speedup vs. numpy |
| --- | --- | --- | --- |
| csuite (numpy) | 1 | 73,100 | 1x |
| catch-jax | 1 | 159,000 | 2.2x |
| catch-jax | 8 | 733,000 | 10x |
| catch-jax | 64 | 2,782,000 | 38x |
| catch-jax | 512 | 5,201,000 | 71x |
| catch-jax | 4096 | 3,679,000 | 50x |

**Agent throughput on catch-jax**

Measured at one seed, so each row is a single agent stepping its own environment
with its own replay buffer. Multi-seed runs scale by vmapping over independent
streams like these rather than by batching environments under one agent.

| Agent | Steps/sec |
| --- | --- |
| Random | 135,000 |
| DQN | 55,000 |
