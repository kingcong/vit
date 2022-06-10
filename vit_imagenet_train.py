# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" ViT training script. """

import argparse

import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from mindvision.classification.dataset import ImageNet
from mindvision.classification.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
from mindvision.engine.callback import LossMonitor
from mindvision.engine.loss import CrossEntropySmooth

set_seed(42)


def vit_train(args_opt):
    """ViT train."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data Pipeline.
    if args_opt.run_distribute:
        init("nccl")
        rank_id = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        dataset_train = ImageNet(args_opt.data_url,
                                 split="train",
                                 shuffle=True,
                                 resize=args_opt.resize,
                                 batch_size=args_opt.batch_size,
                                 repeat_num=args_opt.repeat_num,
                                 num_parallel_workers=args_opt.num_parallel_workers,
                                 num_shards=device_num,
                                 shard_id=rank_id).run()
        ckpt_save_dir = args_opt.ckpt_save_dir + "_ckpt_" + str(rank_id) + "/"
    else:
        dataset_train = ImageNet(args_opt.data_url,
                                 split="train",
                                 shuffle=True,
                                 resize=args_opt.resize,
                                 batch_size=args_opt.batch_size,
                                 repeat_num=args_opt.repeat_num,
                                 num_parallel_workers=args_opt.num_parallel_workers).run()
        ckpt_save_dir = args_opt.ckpt_save_dir

    step_size = dataset_train.get_dataset_size()

    # Create model.
    if args_opt.model == "vit_b_32":
        network = vit_b_32(num_classes=args_opt.num_classes, image_size=args_opt.resize, pretrained=args_opt.pretrained)
    elif args_opt.model == "vit_b_16":
        network = vit_b_16(num_classes=args_opt.num_classes, image_size=args_opt.resize, pretrained=args_opt.pretrained)
    elif args_opt.model == "vit_l_16":
        network = vit_l_16(num_classes=args_opt.num_classes, image_size=args_opt.resize, pretrained=args_opt.pretrained)
    elif args_opt.model == "vit_l_32":
        network = vit_l_32(num_classes=args_opt.num_classes, image_size=args_opt.resize, pretrained=args_opt.pretrained)

    # Set learning rate scheduler.
    if args_opt.lr_decay_mode == "cosine_decay_lr":
        lr = nn.cosine_decay_lr(min_lr=args_opt.min_lr,
                                max_lr=args_opt.max_lr,
                                total_step=args_opt.epoch_size * step_size,
                                step_per_epoch=step_size,
                                decay_epoch=args_opt.decay_epoch)
    elif args_opt.lr_decay_mode == "piecewise_constant_lr":
        lr = nn.piecewise_constant_lr(args_opt.milestone, args_opt.learning_rates)

    # Define optimizer.
    network_opt = nn.Adam(network.trainable_params(), lr, args_opt.momentum)

    # Define loss function.
    network_loss = CrossEntropySmooth(sparse=True,
                                      reduction="mean",
                                      smooth_factor=args_opt.smooth_factor,
                                      classes_num=args_opt.num_classes)

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(save_checkpoint_steps=step_size, keep_checkpoint_max=args_opt.ckpt_max)
    ckpt_callback = ModelCheckpoint(prefix=args_opt.model, directory=ckpt_save_dir, config=ckpt_config)

    # Init the model.
    model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics={"acc"})

    # Begin to train.
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor(lr)],
                dataset_sink_mode=args_opt.dataset_sink_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT train.")

    # TODO @lixaingyi delete some un-useful parameters.
    parser.add_argument("--model", required=False, default="vit_b_16",
                        choices=["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"])
    parser.add_argument("--device_target", type=str, default="GPU", choices=["Ascend", "GPU"])
    parser.add_argument("--data_url", required=True, default=None, help="Location of data.")
    parser.add_argument("--epoch_size", type=int, default=10, help="Train epoch size.")
    parser.add_argument("--pretrained", type=bool, default=False, help="Load pretrained model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of batch size.")
    parser.add_argument("--repeat_num", type=int, default=1, help="Number of repeat.")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of classification.")
    parser.add_argument("--learning_rates", type=list, default=None, help="A list of learning rates.")
    parser.add_argument("--lr_decay_mode", type=str, default="cosine_decay_lr", help="Learning rate decay mode.")
    parser.add_argument("--min_lr", type=float, default=0.0, help="The min learning rate.")
    parser.add_argument("--max_lr", type=float, default=0.003, help="The max learning rate.")
    parser.add_argument("--decay_epoch", type=int, default=90, help="Number of decay epochs.")
    parser.add_argument("--milestone", type=list, default=None, help="A list of milestone.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the moving average.")
    parser.add_argument("--smooth_factor", type=float, default=0.1, help="The smooth factor.")
    parser.add_argument("--resize", type=int, default=224, help="Image resize.")
    parser.add_argument("--ckpt_save_dir", type=str, default="./ViT", help="Location of training outputs.")
    parser.add_argument("--ckpt_max", type=int, default=100, help="Max number of checkpoint files.")
    parser.add_argument("--dataset_sink_mode", type=bool, default=True, help="The dataset sink mode.")
    parser.add_argument("--run_distribute", type=bool, default=True, help="Run distribute.")
    parser.add_argument("--num_parallel_workers", type=int, default=8, help="Number of parallel workers.")

    args = parser.parse_known_args()[0]
    vit_train(args)
