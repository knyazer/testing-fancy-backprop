import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
import timeit

# ==============================================================================
# 1. THE EQUINOX RNN MODEL
# ==============================================================================


class RNNCell(eqx.Module):
    """A simple Recurrent Neural Network cell."""

    W_hy: jax.Array  # Hidden-to-hidden weights
    W_xy: jax.Array  # Input-to-hidden weights
    b: jax.Array  # Bias

    def __init__(self, input_size, hidden_size, *, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.W_hy = jax.random.uniform(key1, (hidden_size, hidden_size))
        self.W_xy = jax.random.uniform(key2, (input_size, hidden_size))
        self.b = jax.random.uniform(key3, (hidden_size,))

    def __call__(self, y_prev, x_t):
        """Perform one step of the RNN."""
        return jnp.tanh(y_prev @ self.W_hy + x_t @ self.W_xy + self.b)


# ==============================================================================
# 2. THE JAX-NATIVE BPTT FUNCTION ADAPTED FOR EQUINOX
# ==============================================================================


def get_bptt_gradients_eqx(model, xs, ys_target):
    """
    Computes stepwise BPTT gradients for an Equinox model.
    """
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


# ==============================================================================
# 3. THE MORE RIGOROUS BENCHMARK
# ==============================================================================

if __name__ == "__main__":
    # --- Benchmark Parameters ---
    # Expanded range of sequence lengths for a clearer view of scaling
    steps_to_test = [10, 20, 50, 100, 200, 400, 800, 1600, 2400, 5000]
    input_size = 32
    hidden_size = 64
    # More runs and repeats for higher stability
    number_of_runs = 20
    number_of_repeats = 7

    key = jax.random.PRNGKey(42)
    model_key, data_key = jax.random.split(key)

    # --- Initialize Model ---
    model = RNNCell(input_size, hidden_size, key=model_key)

    print("=" * 70)
    print("Complex & Rigorous Benchmark: Equinox RNN BPTT")
    print(f"Model: input={input_size}, hidden={hidden_size}")
    print(f"Comparing standard `eqx.filter_grad` vs. our `get_bptt_gradients_eqx`")
    print("=" * 70)
    print(
        f"{'Steps (T)':<10} | {'Baseline (ms)':<15} | {'Stepwise (ms)':<15} | {'Ratio':<10}"
    )
    print("-" * 70)

    # --- Main Benchmark Loop ---
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
        jitted_stepwise_grad_fn = eqx.filter_jit(get_bptt_gradients_eqx)

        # 4. WARM-UP RUNS: ensure JIT compilation is complete before timing
        baseline_grad_fn_warmup = jitted_baseline_grad_fn(model, xs, ys_target)
        stepwise_grad_fn_warmup = jitted_stepwise_grad_fn(model, xs, ys_target)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready(),
            (baseline_grad_fn_warmup, stepwise_grad_fn_warmup),
        )

        # 5. TIME THE FUNCTIONS
        # The lambda now includes blocking to ensure we time the full execution
        baseline_stmt = lambda: jax.tree_util.tree_map(
            lambda x: x.block_until_ready(),
            jitted_baseline_grad_fn(model, xs, ys_target),
        )
        stepwise_stmt = lambda: jax.tree_util.tree_map(
            lambda x: x.block_until_ready(),
            jitted_stepwise_grad_fn(model, xs, ys_target),
        )

        baseline_timer = timeit.Timer(stmt=baseline_stmt)
        stepwise_timer = timeit.Timer(stmt=stepwise_stmt)

        # timeit.repeat runs the timer multiple times and returns a list of results.
        # We take the minimum to get the most stable measurement against system noise.
        baseline_time_sec = (
            min(baseline_timer.repeat(repeat=number_of_repeats, number=number_of_runs))
            / number_of_runs
        )
        stepwise_time_sec = (
            min(stepwise_timer.repeat(repeat=number_of_repeats, number=number_of_runs))
            / number_of_runs
        )

        # 6. PRINT RESULTS
        ratio = stepwise_time_sec / baseline_time_sec
        print(
            f"{steps:<10} | {baseline_time_sec * 1000:<15.4f} | {stepwise_time_sec * 1000:<15.4f} | {ratio:<10.2f}"
        )

    print("=" * 70)
