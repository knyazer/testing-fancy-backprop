"""Sequence copying task using an RNN, as a Problem for gradient-based optimization.

The chain is the sequence of RNN steps: first encoding the input sequence into
the hidden state, then decoding it back out. Without teacher forcing, the
decoding phase uses the RNN's own predictions as input, creating long-range
gradient dependencies proportional to the sequence length.

Note from the paper: the simplified (biased) method with Lyapunov discount is
equivalent to teacher forcing, where
    x_hat_t = f(x_hat_{t-1}, s_{t-1}) * lambda + (1 - lambda) * x_t
Since the loss w.r.t. ground truth is zero, backpropagating through this
recovers exactly the simplified method's formulation.
"""

from typing import Any

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from gradient_based_hpo import Problem, SEED


class RNNCopyCell(eqx.Module):
    """RNN cell with output projection for sequence copying."""

    input_to_hidden: nn.Linear
    hidden_to_hidden: nn.Linear
    hidden_to_output: nn.Linear

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        key: jax.Array,
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.input_to_hidden = nn.Linear(input_size, hidden_size, key=k1)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size, key=k2)
        self.hidden_to_output = nn.Linear(hidden_size, output_size, key=k3)

    def __call__(self, hidden: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        new_hidden = jax.nn.tanh(
            self.input_to_hidden(x) + self.hidden_to_hidden(hidden)
        )
        output_logits = self.hidden_to_output(new_hidden)
        return new_hidden, output_logits


class RNNCopyState(eqx.Module):
    hidden: jax.Array  # (hidden_size,)
    output: jax.Array  # (vocab_size,) logits from last step
    step: jax.Array


class SequenceCopyProblem(Problem):
    rnn_static: Any = eqx.field(static=True)
    initial_rnn_params: Any
    target_sequence: jax.Array  # (seq_length,) integer tokens
    target_onehot: jax.Array  # (seq_length, vocab_size)
    data_key: jax.Array
    max_steps: int = eqx.field(static=True)
    seq_length: int = eqx.field(static=True)
    vocab_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    loss_interval: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        seq_length: int = 20,
        vocab_size: int = 8,
        hidden_size: int = 64,
        key: jax.Array,
        loss_interval: int = 1,
    ):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_steps = 2 * seq_length
        self.loss_interval = loss_interval

        key, rnn_key, seq_key = jr.split(key, 3)

        rnn = RNNCopyCell(vocab_size, hidden_size, vocab_size, key=rnn_key)
        params, static = eqx.partition(rnn, eqx.is_inexact_array)
        self.initial_rnn_params = params
        self.rnn_static = static

        self.target_sequence = jr.randint(seq_key, (seq_length,), 0, vocab_size)
        self.target_onehot = jax.nn.one_hot(
            self.target_sequence, vocab_size, dtype=jnp.float32
        )
        self.data_key = key

    def sample_init_params(self, key: jax.Array):
        """Generate randomly initialized RNN weights."""
        rnn = RNNCopyCell(self.vocab_size, self.hidden_size, self.vocab_size, key=key)
        params, _ = eqx.partition(rnn, eqx.is_inexact_array)
        return params

    def new(self, key: jax.Array | None = None) -> "SequenceCopyProblem":
        """Create a new problem instance with a different data key and target sequence."""
        if key is None:
            key, _ = jr.split(self.data_key)
        seq_key = jr.fold_in(key, 0)
        new_seq = jr.randint(seq_key, (self.seq_length,), 0, self.vocab_size)
        new_onehot = jax.nn.one_hot(new_seq, self.vocab_size, dtype=jnp.float32)
        problem = eqx.tree_at(lambda p: p.data_key, self, key)
        problem = eqx.tree_at(lambda p: p.target_sequence, problem, new_seq)
        problem = eqx.tree_at(lambda p: p.target_onehot, problem, new_onehot)
        return problem

    def initial_state(self, init_params=None):
        """Zero-initialized hidden state and output logits."""
        return RNNCopyState(
            hidden=jnp.zeros(self.hidden_size),
            output=jnp.zeros(self.vocab_size),
            step=jnp.array(0, dtype=jnp.int32),
        )

    def step(self, state: RNNCopyState, params, stepwise_aux):
        """One RNN step: encoding reads ground truth, decoding uses own output."""
        provided_input = (
            stepwise_aux  # (vocab_size,) one-hot during encoding, zeros during decoding
        )

        # During encoding (step < seq_length): use the ground truth token
        # During decoding (step >= seq_length): use own prediction (no teacher forcing)
        is_decoding = state.step >= self.seq_length
        own_prediction = jax.nn.softmax(state.output)
        actual_input = jnp.where(is_decoding, own_prediction, provided_input)

        rnn = eqx.combine(params, self.rnn_static)
        new_hidden, new_output = rnn(state.hidden, actual_input)

        return (
            RNNCopyState(
                hidden=new_hidden,
                output=new_output,
                step=state.step + 1,
            ),
            None,
        )

    def single_step_loss(self, state: RNNCopyState, step_idx=None, step_aux=None):
        """Cross-entropy loss on the RNN output during the decoding phase."""

        def compute_loss():
            target_idx = step_idx - self.seq_length
            target_idx = jnp.clip(target_idx, 0, self.seq_length - 1)
            target = self.target_onehot[target_idx]
            log_probs = jax.nn.log_softmax(state.output)
            return -jnp.sum(target * log_probs) / self.seq_length

        is_decoding = step_idx >= self.seq_length
        should_compute = jnp.logical_and(
            is_decoding,
            step_idx % self.loss_interval == self.loss_interval - 1,
        )
        return jax.lax.cond(should_compute, compute_loss, lambda: 0.0)

    def stepwise_data(self):
        """Per-step inputs: ground truth one-hot during encoding, zeros during decoding."""
        encoding_inputs = self.target_onehot  # (seq_length, vocab_size)
        decoding_inputs = jnp.zeros((self.seq_length, self.vocab_size))
        return jnp.concatenate([encoding_inputs, decoding_inputs], axis=0)


if __name__ == "__main__":
    key = jr.PRNGKey(SEED)

    print("Initializing sequence copy problem...")
    problem = SequenceCopyProblem(seq_length=100, vocab_size=8, hidden_size=32, key=key)
    params = problem.initial_rnn_params

    print(f"Seq length: {problem.seq_length}, Vocab: {problem.vocab_size}")
    print(f"Total steps (encode+decode): {problem.max_steps}")
    print(f"Target: {problem.target_sequence.tolist()}")

    grad_fn = problem.grad(expanded=False)
    jit_grad_fn = eqx.filter_jit(grad_fn)

    print("Computing initial gradient...")
    loss, grads = jit_grad_fn(params)
    grad_norm = jnp.sqrt(sum(float(jnp.sum(g**2)) for g in jax.tree.leaves(grads)))
    print(f"Loss: {float(loss):.4f}, Grad norm: {float(grad_norm):.4e}")

    # Simple training loop: overfit to one sequence
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    print("\nTraining RNN to copy a fixed sequence...")
    for step in range(200):
        loss, grads = jit_grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = jax.tree.map(lambda p, u: p + u, params, updates)
        if step % 20 == 0:
            print(f"  Step {step:3d}: loss={float(loss):.4f}")

    # Test: run forward pass and decode
    rnn = eqx.combine(params, problem.rnn_static)
    hidden = jnp.zeros(problem.hidden_size)
    output = jnp.zeros(problem.vocab_size)

    # Encoding phase
    for i in range(problem.seq_length):
        token_onehot = problem.target_onehot[i]
        hidden, output = rnn(hidden, token_onehot)

    # Decoding phase
    predictions = []
    for i in range(problem.seq_length):
        soft_input = jax.nn.softmax(output)
        hidden, output = rnn(hidden, soft_input)
        predictions.append(int(jnp.argmax(output)))

    print(f"\nTarget:     {problem.target_sequence.tolist()}")
    print(f"Predicted:  {predictions}")
    correct = sum(p == int(t) for p, t in zip(predictions, problem.target_sequence))
    print(f"Accuracy:   {correct}/{problem.seq_length}")
