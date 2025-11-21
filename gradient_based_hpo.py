import argparse
from typing import Any, Tuple

import equinox as eqx
import equinox.nn as nn
import optax
import opts
import jax
import jax.numpy as jnp
import jax.random as jr

from bptt import get_bptt_gradients_fast


class SimpleCNN(eqx.Module):
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    linear: nn.Linear
    height: int = eqx.field(static=True)
    width: int = eqx.field(static=True)

    def __init__(self, height: int, width: int, num_classes: int, *, key: jax.Array):
        k1, k2, k3 = jr.split(key, 3)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            key=k1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            key=k2,
        )
        # each glu halves the hidden dimension
        self.linear = nn.Linear(4 * height * width, num_classes, key=k3)
        self.height = height
        self.width = width

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x[None, ...]  # (28, 28) -> (1, 28, 28)
        x = jax.nn.glu(self.conv1(x))
        x = jax.nn.glu(self.conv2(x))
        x = x.reshape(-1)
        return self.linear(x)


def load_mnist_arrays(num_train: int, num_val: int):
    import tensorflow_datasets as tfds  # type: ignore

    train_ds = tfds.load(
        "mnist", split=f"train[:{num_train}]", as_supervised=True, batch_size=-1
    )
    test_ds = tfds.load(
        "mnist", split=f"test[:{num_val}]", as_supervised=True, batch_size=-1
    )
    train_images, train_labels = tfds.as_numpy(train_ds)
    val_images, val_labels = tfds.as_numpy(test_ds)

    return (
        jnp.asarray(train_images, dtype=jnp.float32),
        jnp.asarray(train_labels, dtype=jnp.int32),
        jnp.asarray(val_images, dtype=jnp.float32),
        jnp.asarray(val_labels, dtype=jnp.int32),
    )


def preprocess_images(images: jax.Array) -> jax.Array:
    images = jnp.asarray(images, dtype=jnp.float32)
    if images.ndim == 4 and images.shape[-1] == 1:
        images = jnp.squeeze(images, axis=-1)
    max_val = jnp.max(images)
    scale = jnp.where(max_val > 1.0, 255.0, 1.0)
    return images / scale


def evaluate_dataset(
    model: SimpleCNN,
    eval_data: tuple[jax.Array],
    *,
    eval_batch_size: int = 128,
) -> Tuple[jax.Array, jax.Array]:
    inputs, targets = eval_data

    def single_loss_fn(input, target):
        return jnp.sum((model(input) - target) ** 2)

    n = (inputs.shape[0] // eval_batch_size) * eval_batch_size

    batched_inputs = inputs[:n].reshape(
        inputs.shape[0] // eval_batch_size, eval_batch_size, *inputs.shape[1:]
    )
    batched_targets = targets[:n].reshape(
        targets.shape[0] // eval_batch_size, eval_batch_size, *targets.shape[1:]
    )

    def batch_body(carry, batch_inputs):
        inputs, targets = batch_inputs
        loss_acc = eqx.filter_vmap(single_loss_fn)(inputs, targets).mean()
        return carry + loss_acc, None

    total_loss, _ = jax.lax.scan(
        batch_body,
        0.0,
        (batched_inputs, batched_targets),
    )
    return total_loss / len(batched_inputs)


def decode_hyperparams(raw_values: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    lr = jnp.exp(raw_values[0])
    if len(raw_values) == 1:
        return lr, 0.9, 1e-5
    momentum = jax.nn.sigmoid(raw_values[1])
    if len(raw_values) == 2:
        return lr, momentum, 1e-5
    reg = jnp.exp(raw_values[2])
    return lr, momentum, reg


def encode_hyperparams(
    lr: float, momentum: float = None, reg: float = None
) -> jax.Array:
    out = [jnp.log(lr)]
    if momentum is not None:
        out.append(jnp.log(momentum) - jnp.log1p(-momentum))
    if reg is not None:
        out.append(jnp.log(reg))
    return jnp.array(out, dtype=jnp.float32)


def train_model(
    raw_hparams: jax.Array,
    init_model: SimpleCNN,
    train_data: tuple[jax.Array],
    batch_size: int,
    num_steps: int,
    key: jax.Array,
):
    train_inputs, train_targets = train_data
    lr, momentum, reg = decode_hyperparams(raw_hparams)
    model_dyn, model_st = eqx.partition(init_model, eqx.is_inexact_array)
    optimizer = opts.sgd(lr, momentum=momentum)
    opt_state = optimizer.init(model_dyn)

    def loss_fn(params, inputs, targets):
        model = eqx.combine(params, model_st)
        return eqx.filter_vmap(lambda x, y: (model(x) - y) ** 2)(inputs, targets).sum()

    def body(carry, key):
        params, opt_state = carry

        idx = jr.choice(key, train_inputs.shape[0], (batch_size,), replace=False)
        grads = eqx.filter_grad(
            lambda p: loss_fn(p, train_inputs[idx], train_targets[idx]),
        )(params)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None

    (model_dyn, _), _ = jax.lax.scan(
        jax.checkpoint(body), (model_dyn, opt_state), jr.split(key, num_steps)
    )
    return eqx.combine(model_dyn, model_st)


def objective_fn(
    raw_hparams: jax.Array,
    init_model: SimpleCNN,
    train_data: tuple[jax.Array],
    val_data: tuple[jax.Array],
    inner_batch_size: int,
    outer_batch_size: int,
    num_steps: int,
    key: jax.Array,
):
    def single_objective(key):
        trained_model = train_model(
            raw_hparams=raw_hparams,
            init_model=init_model,
            train_data=train_data,
            batch_size=inner_batch_size,
            num_steps=num_steps,
            key=key,
        )

        return evaluate_dataset(trained_model, val_data)

    return eqx.filter_vmap(single_objective)(jr.split(key, outer_batch_size)).mean()


def run_hpo(args):
    key = jr.PRNGKey(args.seed)

    data_key, model_key = jr.split(key, 2)
    train_images, train_labels, val_images, val_labels = load_mnist_arrays(
        args.num_train, args.num_val
    )

    num_classes = int(jnp.max(train_labels) - jnp.min(train_labels)) + 1

    train_images = preprocess_images(train_images)
    train_targets = jax.nn.one_hot(train_labels, num_classes, dtype=jnp.float32)

    val_images = preprocess_images(val_images)
    val_targets = jax.nn.one_hot(val_labels, num_classes, dtype=jnp.float32)

    height, width = train_images.shape[1], train_images.shape[2]

    model = SimpleCNN(height, width, num_classes, key=model_key)

    problem = GradientBasedHPO(
        model=model,
        train_data=(train_images, train_targets),
        val_data=(val_images, val_targets),
        num_steps=args.inner_steps,
        batch_size=args.inner_batch,
    )

    outer_params = encode_hyperparams(args.init_lr, args.init_momentum, args.init_reg)
    outer_optimizer = SGD(args.outer_lr)

    outer_objective = eqx.filter_jit(problem.objective)

    for step in range(args.outer_steps):
        val_loss = outer_objective(outer_params)
        updates = outer_optimizer.update(problem, outer_params)
        outer_params = jax.tree.map(lambda p, u: p + u, outer_params, updates)

        lr, momentum, reg = decode_hyperparams(outer_params)
        print(
            f"[outer step {step}] "
            f"val_loss={float(val_loss):.6f} "
            f"lr={float(lr):.5f} reg={float(reg):.6f} momentum={float(momentum):.3f}"
        )

    final_loss = outer_objective(outer_params)
    decoded_params = decode_hyperparams(outer_params)
    print(
        "\nFinal hyperparameters: "
        f"lr={float(decoded_params[0]):.5f}, "
        f"reg={float(decoded_params[2]):.6f}, "
        f"momentum={float(decoded_params[1]):.3f}"
    )
    print(f"Validation metrics: loss={float(final_loss):.6f}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Gradient-based hyperparameter optimization baseline that differentiates "
            "through CNN training dynamics using Equinox."
        )
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=4_000,
        help="number of MNIST samples used to construct the training set",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=1_000,
        help="number of MNIST samples set aside for validation",
    )
    parser.add_argument(
        "--inner-batch",
        type=int,
        default=32,
        help="mini-batch size consumed by each inner SGD step",
    )
    parser.add_argument(
        "--inner-steps",
        type=int,
        default=500,
        help="number of inner optimization iterations per model",
    )
    parser.add_argument(
        "--outer-steps",
        type=int,
        default=1000,
        help="how many outer hyperparameter updates to perform",
    )
    parser.add_argument(
        "--outer-batch",
        type=int,
        default=4,
        help="count of independent inner trainings averaged per outer step",
    )
    parser.add_argument(
        "--outer-lr",
        type=float,
        default=100.0,
        help="learning rate used by the outer optimizer",
    )
    parser.add_argument(
        "--outer-optimizer",
        type=str,
        choices=["sgd", "adam"],
        default="adam",
        help="outer optimizer applied to the raw hyperparameters",
    )
    parser.add_argument(
        "--init-lr", type=float, default=0.0005, help="initial inner learning rate"
    )
    parser.add_argument(
        "--init-reg",
        type=float,
        default=1e-4,
        help="initial weight decay (L2 regularization) scale",
    )
    parser.add_argument(
        "--init-momentum",
        type=float,
        default=0.5,
        help="initial inner momentum hyperparameter",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for all PRNG splits"
    )
    return parser


def build_outer_optimizer(name: str, lr: float) -> optax.GradientTransformation:
    if name == "sgd":
        return optax.sgd(lr)
    if name == "adam":
        return optax.adam(lr)
    raise ValueError(f"Unsupported outer optimizer '{name}'.")


class Problem(eqx.Module):
    max_steps: int = 1000  # the maximum number of steps is forced

    def __init__(self, *args, **kwargs):
        # store static hypers
        raise NotImplementedError

    # defines common methods to interact with the thing in question
    def step(self, state, problem_params):
        # returns (new_state, step_loss)
        # - step_loss is the per-step contribution collected by the scan
        raise NotImplementedError

    def loss(self, step_losses, state):
        # given per-step losses (scan outputs) and final state, what's the scalar loss?
        raise NotImplementedError

    def objective(self, problem_params):
        init_state = self.initial_state()

        def body(carry, _):
            state = carry
            state, step_loss = self.step(state, problem_params)
            return state, step_loss

        rematted_body = jax.checkpoint(body)
        final_state, step_losses = jax.lax.scan(
            rematted_body, init_state, jnp.arange(self.max_steps)
        )
        return self.loss(step_losses, final_state)

    def initial_state(self):
        # generates the initial state based off whatever you want
        raise NotImplementedError


class Optimizer:
    # the interface which kinda isolates the whole optimization logic:
    # since the proposed method requires intimate access to the internals of the problem
    # we are not satisfied with e.g. optax stuff
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, problem, problem_params):
        # returns the update for the parameters
        # e.g. something like:
        # def to_grad(problem_params):
        #   state = problem.initial_state()
        #   for i in range(problem.max_steps):
        #     state, _ = problem.step(state, problem_params)
        #   return problem.loss(state)
        # return -learning_rate * jax.grad(to_grad)(problem_params)
        raise NotImplementedError


class SGD(Optimizer):
    learning_rate: jax.Array

    def __init__(self, learning_rate: float | jax.Array):
        self.learning_rate = jnp.asarray(learning_rate, dtype=jnp.float32)

    def update(self, problem: Problem, problem_params):
        grads = jax.grad(problem.objective)(problem_params)
        return jax.tree.map(lambda g: -self.learning_rate * g, grads)


class TrainState(eqx.Module):
    params: Any
    momentum: Any
    step: jax.Array


class GradientBasedHPO(Problem):
    train_inputs: jax.Array
    train_targets: jax.Array
    val_inputs: jax.Array
    val_targets: jax.Array
    model_static: Any = eqx.field(static=True)
    initial_params: Any
    initial_momentum: Any
    max_steps: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)

    def __init__(
        self,
        model: SimpleCNN,
        train_data: tuple[jax.Array, jax.Array],
        val_data: tuple[jax.Array, jax.Array],
        num_steps: int,
        batch_size: int,
    ):
        params, static = eqx.partition(model, eqx.is_inexact_array)
        self.initial_params = params
        self.initial_momentum = jax.tree.map(jnp.zeros_like, params)
        self.model_static = static
        self.train_inputs, self.train_targets = train_data
        self.val_inputs, self.val_targets = val_data
        self.max_steps = num_steps
        self.batch_size = batch_size

    def _train_loss(self, params, reg_scale, batch_inputs, batch_targets):
        model = eqx.combine(params, self.model_static)
        preds = jax.vmap(model)(batch_inputs)
        data_loss = jnp.mean(jnp.sum((preds - batch_targets) ** 2, axis=-1))

        def l2_sum(tree):
            return jax.tree_util.tree_reduce(
                lambda acc, x: acc + jnp.sum(x**2),
                tree,
                jnp.array(0.0, dtype=jnp.float32),
            )

        reg_loss = reg_scale * l2_sum(params)
        return data_loss + reg_loss

    def _val_loss(self, params):
        model = eqx.combine(params, self.model_static)
        preds = eqx.filter_vmap(model)(self.val_inputs)
        return jnp.mean(jnp.sum((preds - self.val_targets) ** 2, axis=-1))

    def initial_state(self):
        return TrainState(
            params=self.initial_params,
            momentum=self.initial_momentum,
            step=jnp.array(0, dtype=jnp.int32),
        )

    def objective(self, problem_params):
        num_batches = self.train_inputs.shape[0] // self.batch_size
        batched_inputs = self.train_inputs[: num_batches * self.batch_size].reshape(
            num_batches, self.batch_size, *self.train_inputs.shape[1:]
        )
        batched_targets = self.train_targets[: num_batches * self.batch_size].reshape(
            num_batches, self.batch_size, *self.train_targets.shape[1:]
        )

        init_state = self.initial_state()

        def body(state, batch):
            batch_inputs, batch_targets = batch
            return self.step(state, problem_params, batch_inputs, batch_targets), None

        final_state, step_losses = jax.lax.scan(
            jax.checkpoint(body), init_state, (batched_inputs, batched_targets)
        )
        return self.loss(step_losses, final_state)

    def step(self, state, problem_params, batch_inputs, batch_targets):
        lr, momentum, reg = decode_hyperparams(problem_params)

        step_loss, grads = eqx.filter_value_and_grad(
            lambda p: self._train_loss(p, reg, batch_inputs, batch_targets),
        )(state.params)

        new_momentum = jax.tree.map(
            lambda m, g: momentum * m + g, state.momentum, grads
        )
        new_params = jax.tree.map(
            lambda p, m: p - lr * m,
            state.params,
            new_momentum,
        )
        new_step = state.step + 1

        return TrainState(params=new_params, momentum=new_momentum, step=new_step)

    def loss(self, step_losses: jax.Array, state: TrainState):
        return self._val_loss(state.params)


if __name__ == "__main__":
    arg_parser = build_parser()
    run_hpo(arg_parser.parse_args())
