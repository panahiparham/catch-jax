"""Gymnax-style environment Protocol for RL agent typing.

Agents built on the Gymnax compatibility layer accept the tuple-returning
interface described here.  New-style environments implementing
``EnvProtocol`` should be wrapped with ``make_gymnax_compat_env`` before
being passed to any ``make_train`` function.

Adapted from:
https://github.com/andnp/jax-research-template/blob/main/libs/rl-components/src/rl_components/gym_env.py

``step`` differs from the source: it returns separate ``terminated`` and
``truncated`` signals instead of a single combined ``done`` flag.
``GymEnv`` is also marked ``@runtime_checkable`` so conformance can be
verified with ``isinstance`` in tests.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp


class ObservationSpace(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> jnp.dtype: ...


class DiscreteActionSpace(Protocol):
    @property
    def n(self) -> int: ...


class ContinuousActionSpace(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


@runtime_checkable
class GymEnv[ActionSpaceT](Protocol):
    def observation_space(self, params: object | None = None) -> ObservationSpace: ...

    def action_space(self, params: object | None = None) -> ActionSpaceT: ...

    def reset(self, key: jax.Array, params: object | None = None) -> tuple[jax.Array, object]: ...

    def step(
        self,
        key: jax.Array,
        state: Any,
        action: jax.Array,
        params: object | None = None,
    ) -> tuple[jax.Array, object, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]: ...
