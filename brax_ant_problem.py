from typing import Any, Self

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from brax import envs as brax_envs

from problem import Problem


class PolicyMLP(eqx.Module):
    linear1: nn.Linear
    linear2: nn.Linear
    linear3: nn.Linear

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 64,
        *,
        key: jax.Array,
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.linear1 = nn.Linear(obs_size, hidden_size, key=k1)
        self.linear2 = nn.Linear(hidden_size, hidden_size, key=k2)
        self.linear3 = nn.Linear(hidden_size, action_size, key=k3)

    def __call__(self, obs: jax.Array) -> jax.Array:
        x = jax.nn.relu(self.linear1(obs))
        x = jax.nn.relu(self.linear2(x)) + x
        return jax.nn.tanh(self.linear3(x))


class AntSimState(eqx.Module):
    """Simulation state wrapping the Brax environment state."""

    env_state: Any  # brax.envs.base.State (registered JAX pytree)
    step: jax.Array


class BraxAntProblem(Problem):
    env: Any
    policy_static: Any
    initial_policy_params: Any
    data_key: jax.Array
    max_steps: int = eqx.field(static=True)
    obs_size: int = eqx.field(static=True)
    action_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    loss_interval: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_steps: int = 100,
        hidden_size: int = 64,
        key: jax.Array,
        env_name: str = "ant",
        loss_interval: int = 1,
    ):
        env = brax_envs.get_environment(env_name)
        self.env = env
        self.max_steps = num_steps
        self.loss_interval = loss_interval
        self.hidden_size = hidden_size

        key, policy_key, reset_key = jr.split(key, 3)
        reset_state = env.reset(reset_key)
        self.obs_size = int(reset_state.obs.shape[-1])
        self.action_size = int(env.action_size)

        policy = PolicyMLP(
            self.obs_size,
            self.action_size,
            hidden_size=hidden_size,
            key=policy_key,
        )
        params, static = eqx.partition(policy, eqx.is_inexact_array)
        self.initial_policy_params = params
        self.policy_static = static
        self.data_key = key

    def sample_init_params(self, key: jax.Array):
        policy = PolicyMLP(
            self.obs_size,
            self.action_size,
            hidden_size=self.hidden_size,
            key=key,
        )
        params, _ = eqx.partition(policy, eqx.is_inexact_array)
        return params

    def new(self, key: jax.Array | None = None) -> Self:
        if key is None:
            key, _ = jr.split(self.data_key)
        return eqx.tree_at(lambda p: p.data_key, self, key)

    def initial_state(self, init_params=None):
        env_state = self.env.reset(self.data_key)
        return AntSimState(
            env_state=env_state,
            step=jnp.array(0, dtype=jnp.int32),
        )

    def step(self, state: AntSimState, params, stepwise_aux):
        policy = eqx.combine(params, self.policy_static)
        action = policy(state.env_state.obs)
        new_env_state = self.env.step(state.env_state, action)
        return (
            AntSimState(env_state=new_env_state, step=state.step + 1),
            None,
        )

    def single_step_loss(self, state: AntSimState, step_idx=None, step_aux=None):
        def compute_loss():
            return -state.env_state.reward / self.max_steps

        return jax.lax.cond(
            step_idx % self.loss_interval == self.loss_interval - 1,
            compute_loss,
            lambda: 0.0,
        )

    def stepwise_data(self):
        return jnp.zeros((self.max_steps,))


if __name__ == "__main__":
    key = jr.PRNGKey(0)

    print("Initializing Brax Ant problem...")
    problem = BraxAntProblem(key=key)
    params = problem.initial_policy_params

    print(f"Obs size: {problem.obs_size}, Action size: {problem.action_size}")
    print(f"Episode length: {problem.max_steps}")

    grad_fn = problem.grad(expanded=False, windowing=6)
    jit_grad_fn = eqx.filter_jit(grad_fn)

    print("Computing gradient through simulation...")
    loss, grads = jit_grad_fn(params)
    grad_norm = jnp.sqrt(sum(float(jnp.sum(g**2)) for g in jax.tree.leaves(grads)))
    print(f"Loss: {float(loss):.4f}, Grad norm: {float(grad_norm):.4e}")

    # Simple training loop
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    print("\nTraining policy...")
    for step in range(50):
        key, step_key = jr.split(key)
        loss, grads = jit_grad_fn(params, data_key=step_key)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = jax.tree.map(lambda p, u: p + u, params, updates)
        print(f"  Step {step:3d}: loss={float(loss):.4f}")
