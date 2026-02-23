from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Key

NO_WINDOWING = 1_000_000_000


@dataclass(frozen=True)
class ProblemSpec(ABC):
    name: str

    @abstractmethod
    def build(self, *, num_steps: int, key: Key[Array, ""]) -> "Problem":
        ...

    @abstractmethod
    def default_outer_params(self, *, key: Key[Array, ""]) -> Any:
        ...

    def project_outer_params(self, params: Any) -> Any:
        return params

    def describe_outer_params(self, params: Any) -> str:
        return ""

    def sample_init_params(self, *, key: Key[Array, ""]) -> Any | None:
        return None

    def outer_optimizer(self) -> optax.GradientTransformation:
        return optax.sgd(1e-2)


class Problem(eqx.Module):
    max_steps: int = 1000

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def step(self, state, params, stepwise_aux):
        raise NotImplementedError

    def loss(self, params, init_args=None):
        if init_args is None:
            init_state = self.initial_state()
        else:
            init_state = self.initial_state(*init_args)

        def body(state, xs_i):
            stepwise_aux, step_idx = xs_i
            new_state, step_aux = self.step(state, params, stepwise_aux)
            loss = self.single_step_loss(new_state, step_idx=step_idx)
            return new_state, (state, step_aux, loss)

        xs = (self.stepwise_data(), jnp.arange(self.max_steps))
        last_state, (past_states, aux, losses) = jax.lax.scan(body, init_state, xs)

        total_loss = jnp.sum(losses)

        return total_loss

    def initial_state(self, args=None):
        raise NotImplementedError

    def single_step_loss(self, state, step_idx=None, step_aux=None):
        raise NotImplementedError

    def stepwise_data(self):
        raise NotImplementedError

    def grad(
        self,
        windowing: int = NO_WINDOWING,
        expanded: bool = False,
        param_filter=eqx.is_inexact_array,
        ours_simple: bool = False,
        ours_lambda: float = 0.95,
    ):
        if expanded and ours_simple:
            raise ValueError("expanded=True is not compatible with ours_simple=True")

        def fn(
            params: eqx.Module,
            init_args: Any = None,
            data_key: jax.Array | None = None,
        ):
            problem = (
                self
                if data_key is None
                else eqx.tree_at(lambda p: p.data_key, self, data_key)
            )
            params_dyn, params_st = eqx.partition(params, param_filter)
            if init_args is None:
                init_state = problem.initial_state()
            else:
                init_state = problem.initial_state(*init_args)
            state_dyn, state_st = eqx.partition(init_state, eqx.is_array_like)

            xs = (problem.stepwise_data(), jnp.arange(problem.max_steps))
            if not expanded:

                def truncate_state(state, step_idx: int):
                    if ours_simple:

                        def discount(v: jnp.array):
                            return (
                                jax.lax.stop_gradient(v) * (1.0 - ours_lambda)
                                + v * ours_lambda
                            )

                        return jax.tree.map(
                            lambda v: discount(v) if eqx.is_inexact_array(v) else v,
                            state,
                            is_leaf=eqx.is_inexact_array,
                        )
                    return jax.lax.cond(
                        jnp.mod(step_idx, windowing) == 0,
                        lambda s: jax.lax.stop_gradient(s),
                        lambda s: s,
                        state,
                    )

                def loss_body(carry, xs_i):
                    state, loss_acc = carry
                    stepwise_aux, step_idx = xs_i
                    state = truncate_state(state, step_idx=step_idx)
                    new_state, _ = problem.step(
                        eqx.combine(state, state_st), params, stepwise_aux
                    )
                    loss = problem.single_step_loss(new_state, step_idx=step_idx)
                    return (
                        eqx.filter(new_state, eqx.is_array_like),
                        loss_acc + loss,
                    ), None

                loss_body = jax.checkpoint(loss_body)
                init_state_f = eqx.filter(init_state, eqx.is_array_like)
                (_, total_loss), _ = jax.lax.scan(
                    loss_body, (init_state_f, jnp.array(0.0)), xs
                )
                return total_loss

            def body(state, xs_i):
                stepwise_aux, step_idx = xs_i
                if windowing != NO_WINDOWING:
                    state = jax.lax.cond(
                        jnp.mod(step_idx, windowing) == 0,
                        lambda: jax.lax.stop_gradient(state),
                        lambda: state,
                    )
                new_state, step_aux = problem.step(
                    eqx.combine(state, state_st), params, stepwise_aux
                )
                loss = problem.single_step_loss(new_state, step_idx=step_idx)
                return eqx.filter(new_state, eqx.is_array_like), (
                    state,
                    step_aux,
                    loss,
                )

            body = body if expanded else jax.checkpoint(body)
            last_state, (past_states, aux, losses) = jax.lax.scan(
                body, init_state, xs
            )

            total_loss = jnp.sum(losses)

            def backward_body(grad_carry, scan_inputs):
                state, stepwise_aux, step_idx = scan_inputs

                new_state, step_aux = problem.step(
                    eqx.combine(state, state_st),
                    eqx.combine(params_dyn, params_st),
                    stepwise_aux,
                )

                grad_from_loss = eqx.filter_grad(
                    lambda s: problem.single_step_loss(s, step_idx=step_idx)
                )(new_state)
                grad_total = jax.tree.map(
                    lambda g1, g2: g1 + g2, grad_carry, grad_from_loss
                )
                vjp_fn = eqx.filter_vjp(
                    lambda p, s: problem.step(
                        eqx.combine(s, state_st),
                        eqx.combine(p, params_st),
                        stepwise_aux,
                    )[0],
                    params_dyn,
                    state,
                )[1]
                grad_for_params, grad_for_state = vjp_fn(grad_total)

                if windowing != NO_WINDOWING:
                    grad_for_state = jax.lax.cond(
                        jnp.mod(step_idx, windowing) == 0,
                        lambda gs: jax.tree.map(jnp.zeros_like, gs),
                        lambda gs: gs,
                        grad_for_state,
                    )

                return grad_for_state, grad_for_params

            grad_init = jax.tree.map(
                lambda t: jnp.zeros_like(t),
                eqx.filter(init_state, eqx.is_inexact_array),
            )
            _, stepwise_grads_pytree = jax.lax.scan(
                backward_body, grad_init, (past_states, *xs), reverse=True
            )

            stepwise_grads_pytree = jax.tree.map(
                lambda x: x[::-1], stepwise_grads_pytree
            )

            return total_loss, stepwise_grads_pytree

        if expanded:
            return eqx.filter_jit(fn)
        else:
            return eqx.filter_value_and_grad(fn)
