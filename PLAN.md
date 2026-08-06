# Plan: `catch-jax`

A JAX reimplementation of the **continuing Catch** environment from
[google-deepmind/csuite](https://github.com/google-deepmind/csuite)
([`csuite/environments/catch.py`](https://github.com/google-deepmind/csuite/blob/main/csuite/environments/catch.py)),
organized to mirror the [`pinball-jax`](../pinball-jax) project.

Goal: `reset` and `step` are pure functions, fully `jit`- and `vmap`-able, with
fixed-shape state — behaviourally faithful to csuite's Catch, and verified
against a vendored numpy oracle.

---

## 1. The environment

A breakout-like paddle catches falling balls on a `rows × columns` grid
(default 10×5).

| Aspect | csuite behaviour |
| --- | --- |
| Actions | 3 discrete: `LEFT=0`, `STAY=1`, `RIGHT=2`, applied as `dx = action - 1` |
| Paddle | Fixed at row `rows - 1`; `paddle_x` clipped to `[0, columns - 1]` |
| Balls | Fall strictly down, one row per step, in their spawn column |
| Spawning | Each step, with probability `spawn_probability` (default 0.1), one ball appears at row 0 in a uniform-random column |
| Reward | `+1` when a ball reaches the paddle row in the paddle's column, `-1` when it reaches the paddle row elsewhere, `0` otherwise |
| Termination | **Never.** This is a continuing environment |
| Observation | Binary `(rows, columns)` board: 1 where a ball or the paddle sits, 0 elsewhere |

**Step ordering matters** and must be reproduced exactly:

1. Move the paddle (`clip(paddle_x + dx)`).
2. Move every ball down one row.
3. Resolve the ball that landed on the paddle row → reward, then remove it.
4. Maybe spawn a new ball at row 0.
5. Render the observation **after** removal, so a resolved ball is never drawn
   on the paddle row.

### The key invariant: at most one ball per row

All balls descend in lockstep, and at most one ball spawns per timestep, so two
balls can only share a row if two spawned on the same step — which never
happens. Post-`reset` and post-`step`, **each row holds at most one ball**, and
row `rows - 1` is always empty (any ball arriving there is resolved and removed
in the same step, so balls occupy only rows `0 … rows - 2`).

This is what makes a compact, fixed-shape, *exact* (non-truncating) state
representation possible, and it is the crux of the port. It is asserted
explicitly in the test suite.

---

## 2. Confirmed design decisions

| Decision | Choice |
| --- | --- |
| Ball representation | `ball_cols: int32[rows]` + `ball_mask: bool[rows]`, indexed by row |
| Board size | `rows`, `columns` are static constructor args (they determine array shapes) |
| Spawn probability | Lives in `CatchParams`, so it can be swept under `vmap` without retracing |
| Respawn semantics | Match csuite's **code**: probabilistic spawn only (see §3) |
| Discount (benchmark) | `gamma = 0.9` |
| Benchmark metrics | EMA (β=0.99) of reward, and event-gated EMA (β=0.99) of catch rate |
| Warm-up (benchmark) | Updates begin once the buffer holds ≥ `batch` (32) transitions |
| Seeds (benchmark) | 30, as one `jax.vmap` |
| Extras included | `render()` → RGB, GitHub Actions CI, `benchmark_dqn.py` |
| Extras excluded | `example.py` (the README usage snippet covers it) |

---

## 3. Fidelity notes: three issues found in csuite's `catch.py`

These are documented in the module docstring and README so the divergences are
never mistaken for porting bugs.

**(a) The docstring contradicts the code.** The class docstring claims *"A new
ball will always spawn when a ball falls to the bottom of the board,"* but no
such code path exists — the only spawn is the `spawn_probability` draw, which is
independent of whether a ball was just resolved. The board can therefore sit
empty for many consecutive steps. **We implement the code**, so parity testing
is meaningful and results stay comparable to csuite runs.

**(b) `_get_observation` ignores the configured board size.** It allocates
`np.zeros((_ROWS, _COLUMNS))` from the *module constants* rather than
`self._params.rows/columns`, so any non-10×5 board produces an observation whose
shape contradicts the environment's own `observation_spec` (or raises
`IndexError`). **catch-jax uses the real dimensions**, and the vendored oracle
carries the one-line fix so it can be tested at non-default sizes.

**(c) `rows == 1` is silently broken.** With one row, the spawn row and the
paddle row coincide; a ball moves to row 1 (off-board) before the paddle-row
check can ever fire, so it is never resolved and never removed — and rendering
it would index out of bounds. **catch-jax validates `rows >= 2` and
`columns >= 1`** in the constructor with an explicit error.

---

## 4. Deliberate differences from csuite

| csuite | catch-jax | Why |
| --- | --- | --- |
| `start()` / `step()` | `reset(key, params)` / `step(key, state, action, params)` | Conform to the `GymEnv` protocol shared with `pinball-jax` |
| `np.random.Generator` stored **in** `State` | `key` passed to `reset` and `step` | Purity. `step` *consumes* its key (unlike pinball, which discards it) |
| Mutable `State` dataclass with a `list` of balls | Immutable `CatchState` NamedTuple with fixed-shape arrays | JIT/vmap require static shapes |
| `state.paddle_y` | Dropped | Invariantly `rows - 1`; csuite's `set_state` validation exists only to enforce this |
| `get_state()` / `set_state()` | Dropped | A NamedTuple with pure functions is already gettable, settable, and copy-free |
| `observation_spec()` / `action_spec()` (dm_env) | `observation_space()` / `action_space()` | Protocol conformance; no `dm_env` dependency |
| Observation `dtype=int` | `float32` | Matches `pinball-jax`, which casts observations for network consumption |
| No episode cap | `terminated` always `False`; `truncated` when `timestep >= max_steps_in_episode` | The protocol requires both flags; catch is continuing, so only truncation can fire |

The RNG stream cannot be aligned with numpy's — csuite draws the spawn column
*only* when the Bernoulli draw succeeds, whereas JAX splits and draws both
unconditionally. This has no behavioural consequence, but it dictates the
parity-testing strategy in Step 4.

---

## 5. File manifest

```
catch-jax/
├── pyproject.toml                  # hatchling, catch-jax, py>=3.13
├── README.md
├── PLAN.md
├── .gitignore                      # copied from pinball-jax
├── benchmark_dqn.py
├── benchmark_dqn.png / .pdf        # generated
├── .github/workflows/test.yml      # copied from pinball-jax
├── src/catch_jax/
│   ├── __init__.py                 # exports Catch, CatchParams, CatchState
│   ├── catch.py                    # the environment
│   └── gym_env.py                  # copied verbatim from pinball-jax
└── tests/
    ├── conftest.py
    ├── _reference_catch.py         # vendored numpy oracle
    ├── test_catch_protocol.py
    ├── test_catch_dynamics.py
    └── test_catch_parity.py
```

No `configs/` directory (catch has no config files), so `pyproject.toml` also
drops pinball's `artifacts = ["*.cfg"]` line.

---

## Step 1 — Scaffold the repository

- `git init` in `/Users/parhammohammadpanahi/dev/catch-jax`. **No remote is
  added and nothing is pushed** without explicit approval.
- `pyproject.toml`: hatchling backend; `name = "catch-jax"`, `version = "0.1.0"`,
  `requires-python = ">=3.13"`, `dependencies = ["jax"]`;
  `[tool.hatch.build.targets.wheel] packages = ["src/catch_jax"]`;
  `[dependency-groups] dev = ["pytest", "numpy"]`,
  `benchmark = ["optax", "matplotlib", "numpy"]`.
- `src/catch_jax/gym_env.py`: copied **verbatim** from pinball-jax, keeping its
  attribution header (adapted from `andnp/jax-research-template`) so both
  projects share one protocol definition.
- `.gitignore` and `.github/workflows/test.yml`: copied verbatim (uv +
  Python 3.13 + `uv run pytest -v`, on push to `main` and on PRs).
- `uv sync` to create the lockfile and verify the package imports.

**Done when:** `uv run python -c "import catch_jax"` succeeds and
`uv run pytest` collects zero tests without error.

---

## Step 2 — The environment (`src/catch_jax/catch.py`)

Module constants: `DEFAULT_ROWS = 10`, `DEFAULT_COLUMNS = 5`, `NUM_ACTIONS = 3`,
`DEFAULT_SPAWN_PROBABILITY = 0.1`, `DEFAULT_MAX_STEPS_IN_EPISODE`.

```python
class CatchParams(NamedTuple):
    spawn_probability: float = DEFAULT_SPAWN_PROBABILITY
    max_steps_in_episode: int = DEFAULT_MAX_STEPS_IN_EPISODE

class CatchState(NamedTuple):
    paddle_x:  jax.Array   # int32 scalar
    ball_cols: jax.Array   # int32[rows]  — column of the ball in each row
    ball_mask: jax.Array   # bool[rows]   — which rows hold a ball
    timestep:  jax.Array   # int32 scalar
```

`Catch(rows=10, columns=5)` stores `rows`/`columns` as Python ints (static) and
validates `rows >= 2`, `columns >= 1`.

Spaces follow pinball's pattern, except that the observation shape depends on
the board, so `_CatchObservationSpace` is constructed with `(rows, columns)`
rather than reading a module-level constant:

```python
def observation_space(self, params=None) -> ObservationSpace   # .shape == (rows, columns), .dtype == float32
def action_space(self, params=None) -> DiscreteActionSpace     # .n == 3
```

### `reset(key, params=None) -> (obs, state)`

Mirrors csuite's `start`: paddle centred at `columns // 2`, exactly one ball at
row 0 in a uniform-random column, `timestep = 0`.

### `step(key, state, action, params=None) -> (obs, state, reward, terminated, truncated, info)`

In csuite's exact order:

```python
paddle_x = jnp.clip(state.paddle_x + (action - 1), 0, columns - 1)

# 1. Descend. roll() wraps row rows-1 into row 0; that row is always empty by
#    the invariant, but clear index 0 anyway so the spawn slot starts clean.
cols = jnp.roll(state.ball_cols, 1).at[0].set(0)
mask = jnp.roll(state.ball_mask, 1).at[0].set(False)

# 2. Resolve the ball that landed on the paddle row, then remove it.
landed = mask[rows - 1]
reward = jnp.where(landed, jnp.where(cols[rows - 1] == paddle_x, 1.0, -1.0), 0.0)
mask = mask.at[rows - 1].set(False)

# 3. Spawn. Both draws happen unconditionally (see §4).
spawn_key, col_key = jax.random.split(key)
spawn = jax.random.uniform(spawn_key) < params.spawn_probability   # strict <, matching csuite
new_col = jax.random.randint(col_key, (), 0, columns)
cols = cols.at[0].set(jnp.where(spawn, new_col, 0))
mask = mask.at[0].set(spawn)
```

Then `timestep + 1`, `terminated = False` (always — continuing), `truncated =
timestep >= params.max_steps_in_episode`, `info = {}`.

### Observation

One-hot construction, no scatter — an absent ball's placeholder column can
never erase the paddle cell:

```python
balls  = jax.nn.one_hot(cols, columns, dtype=jnp.float32) * mask[:, None]
paddle = jax.nn.one_hot(paddle_x, columns, dtype=jnp.float32) * (jnp.arange(rows) == rows - 1)[:, None]
board  = jnp.maximum(balls, paddle)
```

### `render(state) -> uint8[rows, columns, 3]`

Matches csuite's `common.binary_board_to_rgb` exactly: `board.astype(uint8) *
255`, expanded and tiled to 3 channels — 0 → black, 1 → white. Kept outside the
`GymEnv` protocol as a convenience method (pinball-jax has no `render`, so this
is a deliberate addition).

`__init__.py` exports `Catch`, `CatchParams`, `CatchState`.

**Done when:** the module imports, `Catch().reset(key)` returns a 10×5 float32
board, and `jax.jit`/`jax.vmap` over `step` trace without error.

---

## Step 3 — Behavioural tests

### `tests/test_catch_protocol.py`

- `isinstance(env, GymEnv)`.
- `observation_space().shape == (rows, columns)` and `.dtype == float32`;
  `action_space().n == 3`.
- `Catch()` works with no arguments (unlike `Pinball`, which requires a config).
- `rows=1` and `columns=0` raise `ValueError`.
- `reset`: obs shape/dtype, `CatchState` type, `timestep == 0`, paddle at
  `columns // 2`, exactly one ball and it is at row 0.
- `step` returns the 6-tuple; `timestep` increments; `info == {}`;
  `terminated`/`truncated` are scalar bools.
- `terminated` is `False` for every step of a long rollout (continuing env).
- Truncation fires exactly at `max_steps_in_episode`.

### `tests/test_catch_dynamics.py`

- `LEFT`/`STAY`/`RIGHT` move the paddle by `-1`/`0`/`+1`; clipping holds at both
  walls under sustained pressure.
- With `spawn_probability=0`, balls fall exactly one row per step and no new
  ball ever appears; the board empties after the initial ball resolves.
- Reward is `+1` on a catch, `-1` on a miss, `0` on every other step — driven by
  hand-constructed states for both outcomes.
- A resolved ball is removed the same step and never drawn on the paddle row.
- **Invariant test:** over long rollouts at `spawn_probability = 0.1` and `1.0`,
  every row holds at most one ball, row `rows - 1` is always empty, and at
  `p = 1.0` the steady state holds exactly `rows - 1` balls.
- `p = 1.0` spawns every step; `p = 0.0` never spawns.
- **Statistical RNG tests:** over ~100k steps, the empirical spawn rate matches
  `spawn_probability` and spawned columns are uniform, both within a stated
  tolerance and with fixed seeds so the tests are deterministic.
- Non-default sizes (e.g. 6×3, 2×2, 20×7) behave correctly — the case csuite's
  bug (b) breaks.
- jit/vmap smoke: `jit(step)`; `vmap` over 8 reset keys; `vmap` over
  `spawn_probability` inside `CatchParams` (validating the params split);
  a `lax.scan` rollout.

### `tests/conftest.py`

Minimal: shared fixtures and constants. Notably it does **not** enable
`jax_enable_x64` — pinball-jax needs float64 for floating-point physics parity,
whereas catch's dynamics are integer and exact. The reason is recorded in the
docstring so the omission reads as intentional.

---

## Step 4 — Parity against a vendored numpy oracle

### `tests/_reference_catch.py`

csuite's `catch.py` vendored as a **test-only oracle**, following pinball-jax's
`_reference_pinball.py` convention (clear docstring stating it is a verbatim
reference used only as an oracle, with attribution and the Apache-2.0 header
retained). Modifications, each commented:

1. Drop the `csuite.environments.base` / `common` / `dm_env.specs` imports and
   the base class, so the file stands alone with only numpy.
2. Fix bug (b): allocate the board from `self._params.rows/columns`.
3. Add an injectable spawn source — an optional list of `(spawn: bool, column:
   int)` events consumed one per step in place of the two rng draws. The rng
   path is retained for standalone use.

Dynamics are otherwise untouched.

### `tests/test_catch_parity.py`

Because the two RNG streams cannot be aligned, parity runs in two modes:

**(a) Deterministic parity — `spawn_probability = 0.0`.** Both environments are
fully deterministic given the initial ball column; seed the reference with the
column that JAX's `reset` drew. Roll ≥ 300 random actions and assert
**exact** agreement on the board and the reward at every step, across several
board sizes and seeds.

**(b) Event-replay parity — `p ∈ {0.1, 0.5, 1.0}`.** Roll the JAX env, reading
each step's spawn event directly off `state.ball_mask[0]` / `state.ball_cols[0]`
(unambiguous: post-step, row 0 is either empty or holds exactly the newly
spawned ball). Replay that same event sequence through the reference and assert
exact board and reward agreement over ≥ 500 steps × several seeds. Since the
reference is deterministic once its spawns are fixed, this validates the entire
transition function while letting the two RNGs differ.

Also assert `render()` matches the reference's `binary_board_to_rgb` output
byte-for-byte.

**Done when:** the full suite passes locally and in CI.

---

## Step 5 — `benchmark_dqn.py`

Mirrors pinball-jax's benchmark: one self-contained file, no experiment harness,
DQN vs. a uniform-random baseline, each agent a single `jax.vmap` over 30 seeds.

Hyperparameters from
`NeverEndingRL/experiments/position_catch/Catch50k/DQN.json`:

| Hyperparameter | Value | Source |
| --- | --- | --- |
| Total steps | 50,000 | `total_steps` |
| Episode cutoff | none | `episode_cutoff: -1` → one continuous run |
| Replay buffer | 100,000, uniform | `buffer_size`, `buffer_type: iid` |
| Batch size | 32 | `batch` |
| Update frequency | every 4 steps | `update_freq` |
| Target refresh | hard copy every 128 steps | `target_refresh` |
| Warm-up | 0 → updates start once buffer ≥ 32 | `warmup` (see below) |
| Epsilon | 0.01, constant | `epsilon` |
| Optimizer | Adam, lr 0.01, β₁ 0.9, β₂ 0.999, ε 1e-8 | `optimizer` |
| Network | 2 × 32 ReLU MLP | `representation: TwoLayerRelu, hidden 32` |
| Discount | 0.9 | not in JSON — confirmed choice |
| Seeds | 30 | not in JSON — confirmed choice |

`regularization_type: "None"` / `regularization: 0.0` are inapplicable and
skipped. The observation is the 10×5 board flattened to 50 inputs → 3 outputs.
Since `terminated` is always `False`, every target bootstraps — no terminal
masking. `buffer_size` (100k) exceeds `total_steps` (50k), so the buffer never
evicts.

Two notes on interpretation, both recorded in the file's docstring:

- **Warm-up.** `warmup: 0` is honoured as "no artificial delay," with updates
  gated on the buffer holding ≥ `batch` transitions (≈ step 32), so the first
  updates don't train on 32 copies of the same transition.
- **`position_catch`.** The directory name suggests those experiments may have
  used a position-based observation. catch-jax exposes csuite's binary board, so
  the benchmark uses the board; only the listed hyperparameters are borrowed.

### Metrics

Both are exponential moving averages with β = 0.99, tracked inside the scan:

1. **Reward EMA** — updated every step: `ema = 0.99 * ema + 0.01 * reward`.
2. **Catch-rate EMA** — *event-gated*: updated only on steps where a ball was
   resolved (`reward != 0`), with `1` for a catch and `0` for a miss; unchanged
   on steps where `reward == 0`.

```python
ema_r = 0.99 * ema_r + 0.01 * reward
resolved = reward != 0
ema_c = jnp.where(resolved, 0.99 * ema_c + 0.01 * (reward > 0), ema_c)
```

Both EMAs start at 0 and so are biased low early. Each carries its own update
counter (`n_steps` and `n_resolved`, which differ per seed) and is reported
bias-corrected as `ema / (1 - β**n)`, making the first few thousand steps
readable.

### Figure

Two panels sharing the x-axis — reward EMA on top, catch-rate EMA below — each
showing DQN and random with the mean across 30 seeds and 95% bootstrap
confidence bands. Writes `benchmark_dqn.png` and `benchmark_dqn.pdf`.

Run with `uv run --group benchmark python benchmark_dqn.py`.

**Expected shape of the result** (a sanity check on the implementation, to be
verified when the benchmark actually runs, not asserted in advance): the random
baseline should sit near a catch rate of ~0.2 (uniform ball column over 5
columns) and a reward EMA of ~-0.06 (≈ `0.1 × (0.2 - 0.8)`). Near-perfect play
is reachable — the paddle needs at most 4 moves and has 9 rows of fall time — so
DQN should climb toward a catch rate near 1.0 and a reward EMA near +0.1
(= `spawn_probability`, the resolution rate).

---

## Step 6 — `README.md`

Mirrors pinball-jax's structure:

- One-paragraph description, stating plainly that this is the **continuing**
  csuite Catch (not bsuite's episodic variant) and that `reset`/`step` are pure
  and fully jit/vmap-able.
- **References:** the csuite repository, plus the original Catch citation.
  *Bibliographic details to be verified against the actual sources before
  committing — no citation gets written from memory.*
- **Installation:** `uv add git+https://github.com/<user>/catch-jax` (URL to be
  confirmed).
- **Usage:** a runnable snippet covering `Catch(rows, columns)`,
  `CatchParams(spawn_probability, max_steps_in_episode)`, `reset`, `step`, the
  action encoding, the board observation, and `render()`. This replaces the
  omitted `example.py`, so it includes a short `lax.scan` rollout.
- **Benchmark:** the embedded two-panel plot, the hyperparameter table from
  Step 5, and the run command.
- **Differences from csuite:** the §3 issues and the §4 divergences, so anyone
  comparing against csuite output knows exactly what to expect.

---

## Stated assumptions

Small calls made without asking; each is a one-line change if you disagree.

1. **`DEFAULT_MAX_STEPS_IN_EPISODE` is set very large (`10**9`),** i.e. no
   truncation by default. Catch is continuing and the benchmark config specifies
   `episode_cutoff: -1`, so a small default like pinball's 1000 would silently
   impose an episode boundary the environment does not actually have.
2. **`terminated` is a constant `False` array** rather than being omitted —
   the `GymEnv` protocol requires the field.
3. **Reward dtype is `float32`** and observations are `float32`, matching
   pinball-jax, even though both are integer-valued in csuite.

## Out of scope

- bsuite's *episodic* Catch variant (different dynamics and reward structure).
- `example.py` — folded into the README usage section, per your call.
- Any `git remote` / push / release. Local commits only unless you say otherwise.
