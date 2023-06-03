import time

import torch
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import jit, random, tree_map, value_and_grad, vmap


from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet
from mnist_mlp_train import MLPModel

from flax.training.train_state import TrainState

rngmix = lambda rng, x: random.fold_in(rng, hash(x))

if __name__ == '__main__':
    model = torch.load("/Users/artintajdini/Projects/re-basin-lottery/Lottery-Ticket-Hypothesis-in-Pytorch/saves/fc1/mnist/1_model_lt.pth.tar")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    print(model)
    print(model.classifier[0].weight)

    with wandb.init(
            # project=args.wandb_project,
            # entity=args.wandb_entity,
            project="git-rebasin",
            entity="lottery-re-basin",
            tags=["mnist", "mlp", "training"],
            mode="online",
            job_type="train",
    ) as wandb_run:
        artifact = wandb.Artifact("mnist-lottery-weights", type="model-weights")
        m2 = MLPModel()
        kernel = [None] * 4
        bias = [None] * 4
        for i in range(0, 7, 2):
            print(i)
            kernel[i//2] = model.classifier[i].weight.detach().cpu().numpy()
            bias[i//2] = model.classifier[i].bias.detach().cpu().numpy()

            # [outC, inC] -> [inC, outC]
            kernel[i//2] = jnp.transpose(kernel[i//2], (1, 0))
            # key = random.PRNGKey(0)
            # x = random.normal(key, (1, 3))
        variables = {'params': {'Dense_0': {'kernel': kernel[0], 'bias': bias[0]},
                                'Dense_1': {'kernel': kernel[1], 'bias': bias[1]},
                                'Dense_2': {'kernel': kernel[2], 'bias': bias[2]},
                                'Dense_3': {'kernel': kernel[3], 'bias': bias[3]}}}
        tx = optax.adam(2e-3)
        key = random.PRNGKey(0)
        x = random.normal(key, (28, 28))
        print(m2.apply(variables, x))
        train_state = TrainState.create(
            apply_fn=m2.apply,
            params=variables["params"],  #.init(rngmix(rng, "init"), jnp.zeros((1, 28, 28, 1)))["params"]
            tx=tx,
        )

        with artifact.new_file(f"checkpoint99", mode="wb") as f:
            f.write(flax.serialization.to_bytes(train_state.params))

        wandb_run.log_artifact(artifact)

