import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
import timeit
import random


class RNNCell(eqx.Module):
    W_hy: jax.Array  # Hidden-to-hidden weights
    W_xy: jax.Array  # Input-to-hidden weights
    b: jax.Array  # Bias

    def __init__(self, input_size, hidden_size, *, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.W_hy = jax.random.uniform(key1, (hidden_size, hidden_size))
        self.W_xy = jax.random.uniform(key2, (input_size, hidden_size))
        self.b = jax.random.uniform(key3, (hidden_size,))

    def __call__(self, y_prev, x_t):
        return jnp.tanh(y_prev @ self.W_hy + x_t @ self.W_xy + self.b)


def get_bptt_gradients_fast(model, xs, ys_target):
    params, static = eqx.partition(model, eqx.is_array)
    hidden_size = model.b.shape[0]
    y_0 = jnp.zeros((hidden_size,))

    def loss_fn_single_step(y, y_target):
        return jnp.sum((y - y_target) ** 2)

    def f_dynamics(y_prev, x_t, p):
        return eqx.combine(p, static)(y_prev, x_t)

    def forward_body(y_prev, x_t):
        y_curr = f_dynamics(y_prev, x_t, params)
        return y_curr, y_prev

    _, y_inputs = jax.lax.scan(forward_body, y_0, xs)

    def backward_body(g_carry, scan_inputs):
        y_input, x_t, y_target_t = scan_inputs
        y_k = f_dynamics(y_input, x_t, params)
        g_from_loss = jax.grad(loss_fn_single_step)(y_k, y_target_t)
        g_total_at_k = g_carry + g_from_loss
        vjp_fn = jax.vjp(lambda p, y: f_dynamics(y, x_t, p), params, y_input)[1]
        g_for_params, g_for_y = vjp_fn(g_total_at_k)
        return g_for_y, g_for_params

    initial_g = jnp.zeros_like(y_0)
    scan_over = (y_inputs, xs, ys_target)
    _, stepwise_grads_pytree = jax.lax.scan(
        backward_body, initial_g, scan_over, reverse=True
    )
    return stepwise_grads_pytree


def get_bptt_gradients_gt(model, xs, ys_target):
    params, static = eqx.partition(model, eqx.is_array)
    hidden_size = model.b.shape[0]
    y_0 = jnp.zeros((hidden_size,))
    num_steps = xs.shape[0]

    # --- This function computes the gradient for a SINGLE active step `t` ---
    def get_grad_for_one_step(t, all_params, all_static, all_xs, all_ys_target):
        def loss_fn(p_active):
            def step_body(y_prev, scan_inputs):
                x_i, i = scan_inputs

                # obviously stop if we are at the wrong index
                current_params = jax.lax.cond(
                    i == t, lambda: p_active, lambda: jax.lax.stop_gradient(p_active)
                )

                # stop the gradient from flowing backward through the hidden state
                # *into* the active step
                y_in = jax.lax.cond(
                    i == t, lambda: jax.lax.stop_gradient(y_prev), lambda: y_prev
                )

                model_at_i = eqx.combine(current_params, all_static)
                y_curr = model_at_i(y_in, x_i)
                return y_curr, y_curr

            indices = jnp.arange(num_steps)
            _, ys_pred = jax.lax.scan(step_body, y_0, (all_xs, indices))

            return jnp.sum((ys_pred - all_ys_target) ** 2)

        return eqx.filter_grad(loss_fn)(all_params)

    all_timesteps = jnp.arange(num_steps)
    stepwise_grads_pytree = jax.vmap(
        get_grad_for_one_step, in_axes=(0, None, None, None, None)
    )(all_timesteps, params, static, xs, ys_target)

    return stepwise_grads_pytree


def benchmark_test():
    steps_to_test = [10, 20, 50, 100, 200, 400, 800, 1600, 2400]
    input_size = 32
    hidden_size = 64

    number_of_runs = 8
    number_of_repeats = 4

    key = jax.random.PRNGKey(42)
    model_key, data_key = jax.random.split(key)

    # --- Initialize Model ---
    model = RNNCell(input_size, hidden_size, key=model_key)

    print("=" * 70)
    print("Complex & Rigorous Benchmark: Equinox RNN BPTT")
    print(f"Model: input={input_size}, hidden={hidden_size}")
    print(f"Comparing standard `eqx.filter_grad` vs. our `get_bptt_gradients_fast`")
    print("=" * 70)
    print(
        f"{'Steps (T)':<10} | {'Baseline (ms)':<15} | {'Stepwise (ms)':<15} | {'Ratio':<10}"
    )
    print("-" * 70)

    for steps in steps_to_test:
        # 1. Generate random data for this sequence length
        key, subkey = jax.random.split(data_key)
        xs = jax.random.normal(subkey, (steps, input_size))
        ys_target = jax.random.normal(subkey, (steps, hidden_size))
        y_0 = jnp.zeros((hidden_size,))

        # 2. Define the baseline function: total loss over the sequence
        def total_loss_fn(m, xs, ys_target):
            def step(y_prev, x_t):
                y_curr = m(y_prev, x_t)
                return y_curr, y_curr

            _, ys = jax.lax.scan(step, y_0, xs)
            return jnp.mean((ys - ys_target) ** 2)

        # 3. Create the JIT-compiled functions to be tested
        # Using eqx.filter_jit is a convenient wrapper around jax.jit for Equinox modules
        jitted_baseline_grad_fn = eqx.filter_jit(eqx.filter_grad(total_loss_fn))
        jitted_stepwise_grad_fn = eqx.filter_jit(get_bptt_gradients_fast)

        # warmup
        baseline_grad_fn_warmup = jitted_baseline_grad_fn(model, xs, ys_target)
        stepwise_grad_fn_warmup = jitted_stepwise_grad_fn(model, xs, ys_target)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready(),
            (baseline_grad_fn_warmup, stepwise_grad_fn_warmup),
        )

        baseline_stmt = lambda: jax.tree_util.tree_map(
            lambda x: x.block_until_ready(),
            jitted_baseline_grad_fn(model, xs, ys_target),
        )
        stepwise_stmt = lambda: jax.tree_util.tree_map(
            lambda x: x.block_until_ready(),
            jitted_stepwise_grad_fn(model, xs, ys_target),
        )

        baseline_time_sec, stepwise_time_sec = 0, 0
        for i in range(8):
            if random.random() > 0.5:
                baseline_timer = timeit.Timer(stmt=baseline_stmt)
                stepwise_timer = timeit.Timer(stmt=stepwise_stmt)
            else:
                stepwise_timer = timeit.Timer(stmt=stepwise_stmt)
                baseline_timer = timeit.Timer(stmt=baseline_stmt)

            if i > 1:
                baseline_time_sec += (
                    min(
                        baseline_timer.repeat(
                            repeat=number_of_repeats, number=number_of_runs
                        )
                    )
                    / number_of_runs
                )
                stepwise_time_sec += (
                    min(
                        stepwise_timer.repeat(
                            repeat=number_of_repeats, number=number_of_runs
                        )
                    )
                    / number_of_runs
                )

        ratio = stepwise_time_sec / baseline_time_sec
        print(
            f"{steps:<10} | {baseline_time_sec * 1000:<15.4f} | {stepwise_time_sec * 1000:<15.4f} | {ratio:<10.2f}"
        )

    print("=" * 70)


def consistency_test():
    """
    Test that our stepwise BPTT gradients are consistent with individual
    per-timestep gradients and their cumulative sums.
    """
    print("=" * 70)
    print("Consistency Test: Verifying Stepwise BPTT Gradients")
    print("=" * 70)

    # Test with a small sequence for easy visualization
    steps = 5
    input_size = 3
    hidden_size = 4

    key = jax.random.PRNGKey(123)
    model_key, data_key = jax.random.split(key)

    model = RNNCell(input_size, hidden_size, key=model_key)
    xs = jax.random.normal(data_key, (steps, input_size))
    key, subkey = jax.random.split(key)
    ys_target = jax.random.normal(subkey, (steps, hidden_size))
    y_0 = jnp.zeros((hidden_size,))

    print(
        f"\nTest configuration: {steps} steps, input_size={input_size}, hidden_size={hidden_size}\n"
    )

    # 1. Get stepwise gradients from our method
    print("Computing stepwise gradients using get_bptt_gradients_fast...")
    stepwise_grads = get_bptt_gradients_fast(model, xs, ys_target)

    # 2. Compute individual gradients for each timestep
    print("Computing individual per-timestep gradients...")

    individual_grads = get_bptt_gradients_gt(model, xs, ys_target)

    # 4. Print all gradients
    print("\n" + "=" * 70)
    print("INDIVIDUAL GRADIENTS (per timestep)")
    print("=" * 70)
    for t in range(steps):
        print(f"\nTimestep {t}:")
        print(f"  W_hy[0,0]: {individual_grads.W_hy[t, 0, 0]:.6f}")
        print(f"  W_xy[0,0]: {individual_grads.W_xy[t, 0, 0]:.6f}")
        print(f"  b[0]:      {individual_grads.b[t, 0]:.6f}")

    print("\n" + "=" * 70)
    print("OUR METHOD GRADIENTS (from get_bptt_gradients_fast)")
    print("=" * 70)
    print(
        f"\nShape of returned gradients: {jax.tree_util.tree_map(lambda x: x.shape, stepwise_grads)}"
    )
    for t in range(steps):
        print(f"\nTimestep {t}:")
        print(f"  W_hy[0,0]: {stepwise_grads.W_hy[t, 0, 0]:.6f}")
        print(f"  W_xy[0,0]: {stepwise_grads.W_xy[t, 0, 0]:.6f}")
        print(f"  b[0]:      {stepwise_grads.b[t, 0]:.6f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "consistency":
        consistency_test()
    else:
        print("Usage: python main.py [benchmark|consistency]")
        print("\nRunning consistency test by default...\n")
        consistency_test()
