import argparse
from typing import Any, Tuple

import equinox as eqx
import equinox.nn as nn
import optax
import opts
import jax
import jax.numpy as jnp
import jax.random as jr
from matplotlib import pyplot as plt

# Hyperparameter optimization constants
NUM_TRAIN = 4_000
NUM_VAL = 100
INNER_BATCH = 32
LOSS_INTERVAL = 100
INNER_STEPS = 1000
OUTER_STEPS = 20
OUTER_BATCH = 4
OUTER_LR = 0.005
OUTER_OPTIMIZER = "adam"
INIT_LR = 0.01
INIT_REG = 1e-4
INIT_MOMENTUM = 0.8
SEED = 0


class SimpleCNN(eqx.Module):
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    linear: nn.Linear
    height: int = eqx.field(static=True)
    width: int = eqx.field(static=True)

    def __init__(
        self,
        height: int,
        width: int,
        num_classes: int,
        *,
        key: jax.Array,
        weight_scale: float = 1.0,
    ):
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

        # Scale weights to trigger gradient explosion
        if weight_scale != 1.0:
            self.conv1 = eqx.tree_at(
                lambda m: m.weight, self.conv1, self.conv1.weight * weight_scale
            )
            self.conv2 = eqx.tree_at(
                lambda m: m.weight, self.conv2, self.conv2.weight * weight_scale
            )
            self.linear = eqx.tree_at(
                lambda m: m.weight, self.linear, self.linear.weight * weight_scale
            )

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


def decode_hyperparams(values: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    values = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
    values = jnp.pad(values, (0, max(0, 3 - values.shape[0])))[:3]
    return values[0], values[1], values[2]


def encode_hyperparams(
    lr: float, momentum: float = None, reg: float = None
) -> jax.Array:
    out = [lr]
    if momentum is not None:
        out.append(momentum)
    if reg is not None:
        out.append(reg)
    return jnp.array(out, dtype=jnp.float32)


def project_hyperparams(values: jax.Array) -> jax.Array:
    values = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
    values = jnp.pad(values, (0, max(0, 3 - values.shape[0])))[:3]
    return jnp.array(
        [
            jnp.clip(values[0], 1e-8, 1.0),  # lr
            jnp.clip(values[1], 0.0, 0.999),  # momentum
            jnp.clip(values[2], 1e-8, 1.0),  # reg
        ]
    )


def run_hpo(args):
    key = jr.PRNGKey(SEED)

    data_key, model_key = jr.split(key, 2)
    train_images, train_labels, val_images, val_labels = load_mnist_arrays(
        NUM_TRAIN, NUM_VAL
    )

    num_classes = int(jnp.max(train_labels) - jnp.min(train_labels)) + 1

    train_images = preprocess_images(train_images)
    train_targets = jax.nn.one_hot(train_labels, num_classes, dtype=jnp.float32)

    val_images = preprocess_images(val_images)
    val_targets = jax.nn.one_hot(val_labels, num_classes, dtype=jnp.float32)

    height, width = train_images.shape[1], train_images.shape[2]

    model = SimpleCNN(
        height, width, num_classes, key=model_key, weight_scale=args.weight_scale
    )

    problem = GradientBasedHPO(
        model=model,
        train_data=(train_images, train_targets),
        val_data=(val_images, val_targets),
        num_steps=INNER_STEPS,
        batch_size=INNER_BATCH,
        key=jr.key(0),
        loss_interval=args.loss_interval,
    )

    outer_params = encode_hyperparams(INIT_LR, INIT_MOMENTUM, INIT_REG)

    if OUTER_OPTIMIZER == "adam":
        outer_opt = opts.adam(OUTER_LR)
    else:
        outer_opt = opts.sgd(OUTER_LR, momentum=0.0)

    outer_opt_state = outer_opt.init(outer_params)

    import os

    image_dir = os.path.join("images", args.prefix) if args.prefix else "images"
    os.makedirs(image_dir, exist_ok=True)

    for step in range(OUTER_STEPS):
        val_loss, grads = problem.grad(windowing=args.windowing, expanded=True)(
            outer_params,
            init_params=None,  # set the key here, the env is stoch
        )

        pretty_grads = []
        for i in range(len(grads)):
            norm = jnp.sum(jnp.abs(grads[i]))
            pretty_grads.append(norm)
        pretty_grads = jnp.array(pretty_grads)
        plt.figure()
        plt.plot(pretty_grads)
        plt.savefig(f"{image_dir}/{step}_gradient_norms.png")
        plt.close()

        grads_for_upd = grads.sum(axis=0)
        updates, outer_opt_state = outer_opt.update(
            grads_for_upd, outer_opt_state, outer_params
        )
        decoded_params = decode_hyperparams(outer_params)
        print(
            "\nHyperparameters: "
            f"lr={float(decoded_params[0]):.4f}, "
            f"reg={float(decoded_params[2]):.4f}, "
            f"momentum={float(decoded_params[1]):.3f}\n"
            f"Had Loss Of: {val_loss:.5f}"
        )

        outer_params = jax.tree.map(lambda p, u: p + u, outer_params, updates)
        outer_params = project_hyperparams(outer_params)

    decoded_params = decode_hyperparams(outer_params)
    print(
        "\nFinal hyperparameters: "
        f"lr={float(decoded_params[0]):.5f}, "
        f"reg={float(decoded_params[2]):.6f}, "
        f"momentum={float(decoded_params[1]):.3f}"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Gradient-based hyperparameter optimization baseline that differentiates "
            "through CNN training dynamics using Equinox."
        )
    )
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=1.0,
        help="scale initial network weights to trigger gradient explosion (e.g., 5.0 or 10.0)",
    )
    parser.add_argument(
        "--windowing",
        type=int,
        default=-1,
        help="truncate gradients every N steps (-1 for no truncation)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="subdirectory prefix for saving images (e.g., 'exp1' saves to images/exp1/)",
    )
    parser.add_argument(
        "--loss-interval",
        type=int,
        default=LOSS_INTERVAL,
        help=f"compute validation loss every N steps (default: {LOSS_INTERVAL})",
    )
    return parser


class Problem(eqx.Module):
    max_steps: int = 1000  # the maximum number of steps

    def __init__(self, *args, **kwargs):
        # store constant hypers
        raise NotImplementedError

    # defines common methods to interact with the thing in question
    def step(self, state, params, stepwise_aux):
        # returns new_state, step_aux - with aux being some random data
        raise NotImplementedError

    def loss(self, params, init_args=None):
        # convenience wrapper to get the loss directly
        if init_args is None:
            init_state = self.init_state()
        else:
            init_state = self.init_state(*init_args)

        def body(state, xs_i):
            stepwise_aux, step_idx = xs_i
            new_state, step_aux = self.step(state, params, stepwise_aux)
            loss = self.single_step_loss(new_state, step_idx=step_idx)
            return new_state, (state, aux, loss)

        xs = (self.stepwise_data(), jnp.arange(self.max_steps))
        last_state, (past_states, aux, losses) = jax.lax.scan(body, init_state, xs)

        total_loss = jnp.sum(losses)

        return total_loss

    def initial_state(self, args=None):
        # generates the initial state based off whatever you want
        raise NotImplementedError

    def single_step_loss(self, state, step_idx=None, step_aux=None):
        raise NotImplementedError

    def stepwise_data(self):
        # generates input data for the forward/backward pass, per-step
        # this could be e.g. batches for meta-optimization, character at i^th index
        # for training sentence copying, etc
        raise NotImplementedError

    def grad(
        self,
        windowing: int = -1,
        expanded: bool = False,
        filter=eqx.is_inexact_array,
    ):
        def fn(params: eqx.Module, init_args: Any = None):
            # standard use for init_args: store the random key there
            params_dyn, params_st = eqx.partition(params, filter)
            if init_args is None:
                init_state = self.initial_state()
            else:
                init_state = self.initial_state(*init_args)
            state_dyn, state_st = eqx.partition(init_state, eqx.is_array_like)

            # the loop for scan: a single node in the dependency graph
            def body(state, xs_i):
                stepwise_aux, step_idx = xs_i
                if windowing != -1:
                    state = jax.lax.cond(
                        jnp.mod(step_idx, windowing) == 0,
                        lambda: jax.lax.stop_gradient(state),
                        lambda: state,
                    )
                new_state, step_aux = self.step(
                    eqx.combine(state, state_st), params, stepwise_aux
                )
                loss = self.single_step_loss(new_state, step_idx=step_idx)
                return eqx.filter(new_state, eqx.is_array_like), (state, step_aux, loss)

            xs = (self.stepwise_data(), jnp.arange(self.max_steps))
            body = body if expanded else jax.checkpoint(body)
            last_state, (past_states, aux, losses) = jax.lax.scan(body, init_state, xs)

            # the final loss is the sum of reported losses
            total_loss = jnp.sum(losses)

            # if we are not expanded -> we do filter_and_grad wrap over loss value
            if not expanded:
                return total_loss

            # if we are expanded -> use hand-crafted backprop
            def backward_body(grad_carry, scan_inputs):
                state, stepwise_aux, step_idx = scan_inputs

                new_state, step_aux = self.step(
                    eqx.combine(state, state_st),
                    eqx.combine(params_dyn, params_st),
                    stepwise_aux,
                )

                grad_from_loss = eqx.filter_grad(
                    lambda s: self.single_step_loss(s, step_idx=step_idx)
                )(new_state)
                grad_total = jax.tree.map(
                    lambda g1, g2: g1 + g2, grad_carry, grad_from_loss
                )
                vjp_fn = eqx.filter_vjp(
                    lambda p, s: self.step(
                        eqx.combine(s, state_st),
                        eqx.combine(p, params_st),
                        stepwise_aux,
                    )[0],
                    params_dyn,
                    state,
                )[1]
                grad_for_params, grad_for_state = vjp_fn(grad_total)

                # Apply windowing: zero out grad_for_state AFTER vjp to truncate gradient flow
                # This prevents gradients from flowing to earlier timesteps at window boundaries
                if windowing != -1:
                    grad_for_state = jax.lax.cond(
                        jnp.mod(step_idx, windowing) == 0,
                        lambda gs: jax.tree.map(jnp.zeros_like, gs),
                        lambda gs: gs,
                        grad_for_state,
                    )

                return grad_for_state, grad_for_params

            # initialize the grad with zeroes -> \frac{d_L}{d_s} = 0 (gradient w.r.t. state)
            grad_init = jax.tree.map(
                lambda t: jnp.zeros_like(t),
                eqx.filter(
                    init_state, eqx.is_inexact_array
                ),  # that's what's returned by eqx.filter_grad - inexact arrays
            )
            _, stepwise_grads_pytree = jax.lax.scan(
                backward_body, grad_init, (past_states, *xs), reverse=True
            )

            # invert such that the first value in the vector is 1-long gradient, k-th is k-long gradient
            stepwise_grads_pytree = jax.tree.map(
                lambda x: x[::-1], stepwise_grads_pytree
            )

            return total_loss, stepwise_grads_pytree

        if expanded:
            return eqx.filter_jit(fn)
        else:
            return eqx.filter_value_and_grad(fn)


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
    data_key: jax.Array
    max_steps: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    loss_interval: int = eqx.field(static=True)

    def __init__(
        self,
        model: SimpleCNN,
        train_data: tuple[jax.Array, jax.Array],
        val_data: tuple[jax.Array, jax.Array],
        num_steps: int,
        batch_size: int,
        key: jax.Array,
        loss_interval: int = LOSS_INTERVAL,
    ):
        params, static = eqx.partition(model, eqx.is_inexact_array)
        self.initial_params = params
        self.initial_momentum = jax.tree.map(jnp.zeros_like, params)
        self.model_static = static
        self.train_inputs, self.train_targets = train_data
        self.val_inputs, self.val_targets = val_data
        self.max_steps = num_steps
        self.batch_size = batch_size
        self.loss_interval = loss_interval

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

    def initial_state(self, *_):
        return TrainState(
            params=self.initial_params,
            momentum=self.initial_momentum,
            step=jnp.array(0, dtype=jnp.int32),
        )

    def single_step_loss(self, state: TrainState, step_idx=None, step_aux=None):
        def fn():
            model = eqx.combine(state.params, self.model_static)
            preds = eqx.filter_vmap(model)(self.val_inputs)
            loss = jnp.mean((preds - self.val_targets) ** 2)
            return loss

        return jax.lax.cond(
            step_idx % self.loss_interval == self.loss_interval - 1, fn, lambda: 0.0
        )

    def stepwise_data(self):
        k1, _ = jr.split(self.data_key)
        indices = jr.randint(
            k1, (self.max_steps, self.batch_size), 0, len(self.train_inputs)
        )
        return self.train_inputs[indices], self.train_targets[indices]

    def step(self, state, params, stepwise_aux):
        batch_inputs, batch_targets = stepwise_aux
        lr, momentum, reg = decode_hyperparams(params)

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

        return TrainState(params=new_params, momentum=new_momentum, step=new_step), None


if __name__ == "__main__":
    arg_parser = build_parser()
    run_hpo(arg_parser.parse_args())
