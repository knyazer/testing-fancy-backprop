from typing import Any

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from problem import Problem


class SimpleMLP(eqx.Module):
    linear1: nn.Linear
    linear2: nn.Linear
    linear3: nn.Linear
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
        in_dim = height * width
        hidden = 32
        self.linear1 = nn.Linear(in_dim, hidden, key=k1)
        self.linear2 = nn.Linear(hidden, hidden, key=k2)
        self.linear3 = nn.Linear(hidden, num_classes, key=k3)
        self.height = height
        self.width = width

        if weight_scale != 1.0:
            self.linear1 = eqx.tree_at(
                lambda m: m.weight, self.linear1, self.linear1.weight * weight_scale
            )
            self.linear2 = eqx.tree_at(
                lambda m: m.weight, self.linear2, self.linear2.weight * weight_scale
            )
            self.linear3 = eqx.tree_at(
                lambda m: m.weight, self.linear3, self.linear3.weight * weight_scale
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape(-1)
        x = jax.nn.relu(self.linear1(x))
        x = jax.nn.relu(self.linear2(x))
        return self.linear3(x)


def load_mnist_arrays(
    num_train: int, num_val: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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


def decode_hyperparams(values: jax.Array) -> tuple[jax.Array]:
    values = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
    values = jnp.pad(values, (0, max(0, 1 - values.shape[0])))[:1]
    return (values[0],)


def encode_hyperparams(lr: float) -> jax.Array:
    return jnp.array([lr], dtype=jnp.float32)


def project_hyperparams(values: jax.Array) -> jax.Array:
    values = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
    values = jnp.pad(values, (0, max(0, 1 - values.shape[0])))[:1]
    return jnp.array([jnp.clip(values[0], 1e-8, 1.0)])


class TrainState(eqx.Module):
    params: Any
    step: jax.Array


class GradientBasedHPO(Problem):
    train_inputs: jax.Array
    train_targets: jax.Array
    val_inputs: jax.Array
    val_targets: jax.Array
    model_static: Any = eqx.field(static=True)
    initial_params: Any
    data_key: jax.Array
    max_steps: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    weight_scale: float = eqx.field(static=True)
    loss_interval: int = eqx.field(static=True)

    def __init__(
        self,
        model: SimpleMLP,
        train_data: tuple[jax.Array, jax.Array],
        val_data: tuple[jax.Array, jax.Array],
        num_steps: int,
        batch_size: int,
        key: jax.Array,
        weight_scale: float = 1.0,
        loss_interval: int = 1,
    ):
        params, static = eqx.partition(model, eqx.is_inexact_array)
        self.initial_params = params
        self.model_static = static
        self.train_inputs, self.train_targets = train_data
        self.val_inputs, self.val_targets = val_data
        self.data_key = key
        self.max_steps = num_steps
        self.batch_size = batch_size
        self.weight_scale = weight_scale
        self.loss_interval = loss_interval

    def sample_init_params(self, key: jax.Array):
        height, width = self.train_inputs.shape[1], self.train_inputs.shape[2]
        num_classes = self.train_targets.shape[-1]
        model = SimpleMLP(
            height,
            width,
            num_classes,
            key=key,
            weight_scale=self.weight_scale,
        )
        params, _ = eqx.partition(model, eqx.is_inexact_array)
        return params

    def new(self, key: jax.Array | None = None) -> "GradientBasedHPO":
        if key is None:
            key, _ = jr.split(self.data_key)
        return eqx.tree_at(lambda p: p.data_key, self, key)

    def _train_loss(self, params, batch_inputs, batch_targets):
        model = eqx.combine(params, self.model_static)
        preds = jax.vmap(model)(batch_inputs)
        loss = optax.softmax_cross_entropy(preds, batch_targets)
        return jnp.mean(loss)

    def initial_state(self, init_params=None):
        if init_params is None:
            init_params = self.initial_params
        return TrainState(
            params=init_params,
            step=jnp.array(0, dtype=jnp.int32),
        )

    def single_step_loss(self, state: TrainState, step_idx=None, step_aux=None):
        def fn():
            model = eqx.combine(state.params, self.model_static)
            preds = eqx.filter_vmap(model)(self.val_inputs)
            loss = optax.softmax_cross_entropy(preds, self.val_targets)
            return jnp.mean(loss) / self.max_steps

        return jax.lax.cond(
            step_idx % self.loss_interval == self.loss_interval - 1, fn, lambda: 0.0
        )

    def stepwise_data(self):
        indices = jr.randint(
            self.data_key, (self.max_steps, self.batch_size), 0, len(self.train_inputs)
        )
        return self.train_inputs[indices], self.train_targets[indices]

    def step(self, state, params, stepwise_aux):
        batch_inputs, batch_targets = stepwise_aux
        (lr,) = decode_hyperparams(params)

        _step_loss, grads = eqx.filter_value_and_grad(
            lambda p: self._train_loss(p, batch_inputs, batch_targets),
        )(state.params)

        new_params = jax.tree.map(
            lambda p, g: p - lr * g,
            state.params,
            grads,
        )
        new_step = state.step + 1

        return TrainState(params=new_params, step=new_step), None
