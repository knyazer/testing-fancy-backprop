import argparse
from typing import Any, Tuple

import equinox as eqx
import equinox.nn as nn
import optax
import opts
import jax
import jax.numpy as jnp
import jax.random as jr


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


def decode_hyperparams(raw_values: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    raw_values = jnp.asarray(raw_values, dtype=jnp.float32).reshape(-1)
    raw_values = jnp.nan_to_num(raw_values, nan=0.0, posinf=20.0, neginf=-20.0)

    num_values = int(min(raw_values.shape[0], 3))
    raw_values = raw_values[:num_values]
    raw_values = jnp.pad(raw_values, (0, 3 - num_values))

    lr = jnp.exp(raw_values[0])
    momentum = jax.nn.sigmoid(raw_values[1])
    reg = jnp.exp(raw_values[2])

    if num_values == 1:
        return lr, 0.9, 1e-5
    if num_values == 2:
        return lr, momentum, 1e-5
    return lr, momentum, reg


def encode_hyperparams(
    lr: float, momentum: float = None, reg: float = None
) -> jax.Array:
    tiny = jnp.float32(1e-12)
    lr_safe = jnp.clip(jnp.asarray(lr, dtype=jnp.float32), tiny, None)

    out = [jnp.log(lr_safe)]
    if momentum is not None:
        momentum_safe = jnp.clip(
            jnp.asarray(momentum, dtype=jnp.float32), tiny, 1.0 - tiny
        )
        out.append(jnp.log(momentum_safe) - jnp.log1p(-momentum_safe))
    if reg is not None:
        reg_safe = jnp.clip(jnp.asarray(reg, dtype=jnp.float32), tiny, None)
        out.append(jnp.log(reg_safe))
    return jnp.array(out, dtype=jnp.float32)


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
        key=jr.key(0),
    )

    outer_params = encode_hyperparams(args.init_lr, args.init_momentum, args.init_reg)
    outer_optimizer = SGD(args.outer_lr)

    for step in range(args.outer_steps):
        v1, g1 = problem.grad()(outer_params)
        v3, g3 = problem.grad(windowing=10)(outer_params)
        v4, g4 = problem.grad(windowing=10, expanded=True)(outer_params)
        print(g4.sum(axis=0), "expanded")
        print("should be equal to ", g3)
        print("should _not_ be equal to", g1)
        breakpoint()

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
        self, windowing: int = -1, expanded: bool = False, filter=eqx.is_inexact_array
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
                        jnp.mod(step_idx, windowing) == windowing - 1,
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
                        jnp.mod(step_idx, windowing) == windowing - 1,
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
                ),  # that's what's returned by eqx.filter_grad
            )
            _, stepwise_grads_pytree = jax.lax.scan(
                backward_body, grad_init, (past_states, *xs), reverse=True
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

    def __init__(
        self,
        model: SimpleCNN,
        train_data: tuple[jax.Array, jax.Array],
        val_data: tuple[jax.Array, jax.Array],
        num_steps: int,
        batch_size: int,
        key: jax.Array,
    ):
        params, static = eqx.partition(model, eqx.is_inexact_array)
        self.initial_params = params
        self.initial_momentum = jax.tree.map(jnp.zeros_like, params)
        self.model_static = static
        self.train_inputs, self.train_targets = train_data
        self.val_inputs, self.val_targets = val_data
        self.max_steps = num_steps
        self.batch_size = batch_size
        self.data_key = key

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
        model = eqx.combine(state.params, self.model_static)
        preds = eqx.filter_vmap(model)(self.val_inputs[:100])
        loss = jnp.mean((preds - self.val_targets[:100]) ** 2)
        return jax.lax.cond(step_idx == self.max_steps - 1, lambda: loss, lambda: 0.0)

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


class Optimizer:
    # the interface which kinda isolates the whole optimization logic:
    # since the proposed method requires intimate access to the internals of the problem
    # we are not satisfied with e.g. optax stuff
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, problem, problem_params):
        raise NotImplementedError


class SGD(Optimizer):
    learning_rate: jax.Array

    def __init__(self, learning_rate: float | jax.Array):
        self.learning_rate = jnp.asarray(learning_rate, dtype=jnp.float32)

    def update(self, problem: Problem, problem_params):
        grads = jax.grad(problem.objective)(problem_params)
        return jax.tree.map(lambda g: -self.learning_rate * g, grads)


if __name__ == "__main__":
    arg_parser = build_parser()
    run_hpo(arg_parser.parse_args())
