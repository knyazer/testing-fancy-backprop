import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Any


class SGDState(eqx.Module):
    momentum: Any  # PyTree matching parameter structure


class AdamWState(eqx.Module):
    m: Any  # First moment estimate
    v: Any  # Second moment estimate
    step: jax.Array


class Optimizer(eqx.Module):
    def __init__(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class SGD(eqx.Module):
    learning_rate: jax.Array
    momentum: jax.Array

    def __init__(
        self, learning_rate: float | jax.Array, momentum: float | jax.Array = 0.9
    ):
        self.learning_rate = jnp.asarray(learning_rate, dtype=jnp.float32)
        self.momentum = jnp.asarray(momentum, dtype=jnp.float32)

    def init(self, params: Any) -> SGDState:
        return SGDState(momentum=jax.tree.map(jnp.zeros_like, params))

    def update(
        self, grads: Any, state: SGDState, params: Any | None = None
    ) -> tuple[Any, SGDState]:
        new_m = jax.tree.map(lambda m, g: self.momentum * m + g, state.momentum, grads)
        updates = jax.tree.map(lambda m: -self.learning_rate * m, new_m)
        return updates, SGDState(momentum=new_m)


class AdamW(eqx.Module):
    learning_rate: jax.Array
    b1: jax.Array
    b2: jax.Array
    eps: jax.Array
    weight_decay: jax.Array

    def __init__(
        self,
        learning_rate: float | jax.Array,
        b1: float | jax.Array = 0.9,
        b2: float | jax.Array = 0.999,
        eps: float | jax.Array = 1e-8,
        weight_decay: float | jax.Array = 0.0001,
    ):
        self.learning_rate = jnp.asarray(learning_rate, dtype=jnp.float32)
        self.b1 = jnp.asarray(b1, dtype=jnp.float32)
        self.b2 = jnp.asarray(b2, dtype=jnp.float32)
        self.eps = jnp.asarray(eps, dtype=jnp.float32)
        self.weight_decay = jnp.asarray(weight_decay, dtype=jnp.float32)

    def init(self, params: Any) -> AdamWState:
        m = jax.tree.map(jnp.zeros_like, params)
        v = jax.tree.map(jnp.zeros_like, params)
        return AdamWState(m=m, v=v, step=jnp.array(0, dtype=jnp.int32))

    def update(
        self, grads: Any, state: AdamWState, params: Any
    ) -> tuple[Any, AdamWState]:
        step = state.step + 1
        new_m = jax.tree.map(
            lambda m, g: self.b1 * m + (1 - self.b1) * g, state.m, grads
        )
        new_v = jax.tree.map(
            lambda v, g: self.b2 * v + (1 - self.b2) * g**2, state.v, grads
        )

        step_float = step.astype(jnp.float32)
        b1_power = jnp.exp(step_float * jnp.log(self.b1))
        b2_power = jnp.exp(step_float * jnp.log(self.b2))

        m_hat_scale = 1.0 / (1.0 - b1_power)
        v_hat_scale = 1.0 / (1.0 - b2_power)

        updates = jax.tree.map(
            lambda m, v, p: -self.learning_rate
            * (m * m_hat_scale)
            / (jnp.sqrt(v * v_hat_scale) + self.eps)
            - self.weight_decay * p,
            new_m,
            new_v,
            params,
        )

        return updates, AdamWState(m=new_m, v=new_v, step=step)


def sgd(learning_rate: float | jax.Array, momentum: float | jax.Array = 0.9) -> SGD:
    return SGD(learning_rate, momentum)


def adam(
    learning_rate: float | jax.Array,
    b1: float | jax.Array = 0.9,
    b2: float | jax.Array = 0.999,
    eps: float | jax.Array = 1e-8,
) -> AdamW:
    """Create an AdamW optimizer."""
    return AdamW(learning_rate, b1, b2, eps, 0.0)


def adamw(
    learning_rate: float | jax.Array,
    b1: float | jax.Array = 0.9,
    b2: float | jax.Array = 0.999,
    eps: float | jax.Array = 1e-8,
    weight_decay: float | jax.Array = 0.01,
) -> AdamW:
    """Create an AdamW optimizer."""
    return AdamW(learning_rate, b1, b2, eps, weight_decay)
