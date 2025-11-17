import argparse
from typing import Tuple

import equinox as eqx
import equinox.nn as nn
import optax
import jax
import jax.numpy as jnp
import jax.random as jr

from main import get_bptt_gradients_fast


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
        self.linear = nn.Linear(16 * height * width, num_classes, key=k3)
        self.height = height
        self.width = width

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x[None, ...]  # (28, 28) -> (1, 28, 28)
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
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
        loss_acc = jax.vmap(single_loss_fn)(inputs, targets).mean()
        return carry + loss_acc, None

    total_loss, _ = jax.lax.scan(
        batch_body,
        0.0,
        (batched_inputs, batched_targets),
    )
    return total_loss / len(batched_inputs)


def decode_hyperparams(raw_values: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    lr = jnp.exp(raw_values[0])
    reg = jnp.exp(raw_values[1])
    momentum = jax.nn.sigmoid(raw_values[2])
    return lr, reg, momentum


def encode_hyperparams(lr: float, reg: float, momentum: float) -> jax.Array:
    log_lr = jnp.log(lr)
    log_reg = jnp.log(reg)
    logit_m = jnp.log(momentum) - jnp.log1p(-momentum)
    return jnp.array([log_lr, log_reg, logit_m], dtype=jnp.float32)


def train_model(
    raw_hparams: jax.Array,
    init_model: SimpleCNN,
    train_data: tuple[jax.Array],
    batch_size: int,
    num_steps: int,
    key: jax.Array,
):
    train_inputs, train_targets = train_data
    lr, reg, momentum = decode_hyperparams(raw_hparams)
    model_dyn, model_st = eqx.partition(init_model, eqx.is_inexact_array)
    optimizer = optax.adamw(lr, b1=momentum, weight_decay=reg)
    opt_state = optimizer.init(model_dyn)

    def loss_fn(model, inputs, targets):
        return eqx.filter_vmap(lambda x, y: (model(x) - y) ** 2)(inputs, targets).sum()

    def body(carry, key):
        params, opt_state = carry

        idx = jr.choice(key, train_inputs.shape[0], (batch_size,), replace=False)
        grads = eqx.filter_grad(
            lambda m: loss_fn(m, train_inputs[idx], train_targets[idx]),
        )(eqx.combine(params, model_st))

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), None

    checkpointed_body = jax.checkpoint(body)
    (model_dyn, _), _ = jax.lax.scan(
        checkpointed_body, (model_dyn, opt_state), jr.split(key, num_steps)
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

    data_key, model_key, objective_key = jr.split(key, 3)
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

    outer_params = encode_hyperparams(args.init_lr, args.init_reg, args.init_momentum)
    outer_optimizer = build_outer_optimizer(args.outer_optimizer, args.outer_lr)
    outer_opt_state = outer_optimizer.init(outer_params)

    wrapped_objective_fn = eqx.Partial(
        objective_fn,
        outer_batch_size=args.outer_batch,
        inner_batch_size=args.inner_batch,
        init_model=model,
        train_data=(train_images, train_targets),
        val_data=(val_images, val_targets),
        num_steps=args.inner_steps,
        key=objective_key,
    )

    for step in range(args.outer_steps):
        val_loss, grad = eqx.filter_value_and_grad(wrapped_objective_fn)(outer_params)
        updates, outer_opt_state = outer_optimizer.update(
            grad, outer_opt_state, outer_params
        )
        outer_params = optax.apply_updates(outer_params, updates)

        lr, reg, momentum = outer_params
        print(
            f"[outer step {step}] "
            f"val_loss={float(val_loss):.6f} "
            f"lr={float(lr):.5f} reg={float(reg):.6f} momentum={float(momentum):.3f}"
        )

    final_loss = wrapped_objective_fn(outer_params)
    print(
        "\nFinal hyperparameters: "
        f"lr={float(outer_params[0]):.5f}, "
        f"reg={float(outer_params[1]):.6f}, "
        f"momentum={float(outer_params[2]):.3f}"
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
        default=40_000,
        help="number of MNIST samples used to construct the training set",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=10_000,
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
        default=40,
        help="number of inner optimization iterations per model",
    )
    parser.add_argument(
        "--outer-steps",
        type=int,
        default=4,
        help="how many outer hyperparameter updates to perform",
    )
    parser.add_argument(
        "--outer-batch",
        type=int,
        default=1,
        help="count of independent inner trainings averaged per outer step",
    )
    parser.add_argument(
        "--outer-lr",
        type=float,
        default=10.0,
        help="learning rate used by the outer optimizer",
    )
    parser.add_argument(
        "--outer-optimizer",
        type=str,
        choices=["sgd", "adam"],
        default="sgd",
        help="outer optimizer applied to the raw hyperparameters",
    )
    parser.add_argument(
        "--init-lr", type=float, default=0.01, help="initial inner learning rate"
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


if __name__ == "__main__":
    arg_parser = build_parser()
    run_hpo(arg_parser.parse_args())
