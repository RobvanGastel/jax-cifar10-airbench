"""
A copy of airbench94_muon.py to train JAX models for CIFAR10.
"""

#############################################
#                  Setup                    #
#############################################

import os
import time
from math import ceil
from typing import Any

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import jax
import optax
import flax.linen as nn
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.training import train_state
from flax.training import train_state, checkpoints


#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    images_out = torch.empty(
        (len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype
    )
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[
                    mask, :, r + sy : r + sy + crop_size, r + sx : r + sx + crop_size
                ]
    else:
        images_tmp = torch.empty(
            (len(images), 3, crop_size, crop_size + 2 * r),
            device=images.device,
            dtype=images.dtype,
        )
        for s in range(-r, r + 1):
            mask = shifts[:, 0] == s
            images_tmp[mask] = images[mask, :, r + s : r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = shifts[:, 1] == s
            images_out[mask] = images_tmp[mask, :, :, r + s : r + s + crop_size]
    return images_out


class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save(
                {"images": images, "labels": labels, "classes": dset.classes}, data_path
            )

        data = torch.load(data_path, map_location=torch.device("cuda"))
        self.images, self.labels, self.classes = (
            data["images"],
            data["labels"],
            data["classes"],
        )
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (
            (self.images.half() / 255)
            .permute(0, 3, 1, 2)
            .to(memory_format=torch.channels_last)
        )

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = (
            {}
        )  # Saved results of image processing to be done on the first epoch
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ["flip", "translate"], "Unrecognized key: %s" % k

        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return (
            len(self.images) // self.batch_size
            if self.drop_last
            else ceil(len(self.images) / self.batch_size)
        )

    def __iter__(self):

        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get("flip", False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(
            len(images), device=images.device
        )
        for i in range(len(self)):
            idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield (images[idxs], self.labels[idxs])


#############################################
#            Network Definition             #
#############################################


class BasicBlock(nn.Module):
    features: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool):
        residual = x

        x = nn.Conv(self.features, (3, 3), self.stride, padding="SAME", use_bias=False)(
            x
        )
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5)(x)
        x = nn.relu(x)

        x = nn.Conv(self.features, (3, 3), 1, padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5)(x)

        if residual.shape != x.shape:
            residual = nn.Conv(self.features, (1, 1), self.stride, use_bias=False)(
                residual
            )
            residual = nn.BatchNorm(
                use_running_average=not train, momentum=0.9, epsilon=1e-5
            )(residual)

        return nn.relu(x + residual)


class ResNet18(nn.Module):
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, train: bool):
        # CIFAR-style stem
        x = nn.Conv(64, (3, 3), 1, padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # 2,2,2,2 blocks
        for features, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            x = BasicBlock(features, stride)(x, train)
            x = BasicBlock(features, 1)(x, train)

        x = jnp.mean(x, axis=(1, 2))  # global average pool
        x = nn.Dense(self.num_classes)(x)
        return x


############################################
#                 Logging                  #
############################################


def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ""
    for col in columns_list:
        print_string += "|  %s  " % col
    print_string += "|"
    if is_head:
        print("-" * len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print("-" * len(print_string))


logging_columns_list = [
    "run   ",
    "epoch",
    "train_acc",
    "val_acc",
    "tta_val_acc",
    "time_seconds",
]


def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = "{:0.4f}".format(var)
        else:
            assert var is None
            res = ""
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)


############################################
#               Evaluation                 #
############################################

# TODO:

############################################
#                Training                  #
############################################


class TrainState(train_state.TrainState):
    batch_stats: FrozenDict[str, Any]


def main(run, model):
    batch_size = 2000
    wd = 2e-6 * batch_size

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader(
        "cifar10",
        train=True,
        batch_size=batch_size,
        aug=dict(flip=True, translate=2),
    )

    total_train_steps = ceil(20 * len(train_loader))

    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, jnp.ones((1, 32, 32, 3)), train=True)
    tx = optax.adamw(learning_rate=0.001, weight_decay=wd)

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables.get("batch_stats", {}),
    )

    @jax.jit
    def train_step(state, images, labels):
        def loss_fn(params):
            logits, new_model_state = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                images,
                train=True,
                mutable=["batch_stats"],
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=labels
            ).mean()
            return loss, new_model_state

        (loss, new_model_state), grads = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=new_model_state["batch_stats"])
        return state, loss

    @jax.jit
    def eval_step(state, images, labels):
        logits, _ = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats},
            images,
            train=False,
            mutable=["batch_stats"],
        )
        preds = jnp.argmax(logits, axis=-1)
        return jnp.mean(preds == labels)

    # Training code
    start = time.perf_counter()
    for epoch in range(ceil(total_train_steps / len(train_loader))):
        for images, labels in train_loader:
            images = jnp.array(images.cpu().numpy()).transpose(0, 2, 3, 1)
            labels = jnp.array(labels.cpu().numpy())

            state, loss = train_step(state, images, labels)
    end = time.perf_counter()

    checkpoints.save_checkpoint(
        ckpt_dir=f"model_0",
        target=state,
        step=epoch,
        overwrite=True,
    )

    print(f"Training time: {end - start}s")

    start = time.perf_counter()
    accs = []
    for images, labels in test_loader:
        images = jnp.array(images.cpu().numpy()).transpose(0, 2, 3, 1)
        labels = jnp.array(labels.cpu().numpy())
        accs.append(eval_step(state, images, labels))
    end = time.perf_counter()

    print(f"Validation time: {end - start:.3f}s, {jnp.mean(jnp.array(accs)):.3f}")


if __name__ == "__main__":

    model = ResNet18()
    main("warmup", model)

    # TODO: Implement the next steps
