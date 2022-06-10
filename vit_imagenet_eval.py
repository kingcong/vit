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
"""eval script"""

import argparse

import mindspore.nn as nn
from mindspore import context
from mindspore.train import Model

from mindvision.classification.dataset import ImageNet
from mindvision.classification.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
from mindvision.engine.loss import CrossEntropySmooth


def vit_eval(args_opt):
    """vision transformer eval."""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data pipeline.
    dataset_path = args_opt.data_url
    dataset = ImageNet(dataset_path,
                       split="val",
                       num_parallel_workers=args_opt.num_parallel_workers,
                       resize=args_opt.resize,
                       batch_size=args_opt.batch_size)
    dataset_eval = dataset.run()

    # Create model.
    if args_opt.model == "vit_b_32":
        network = vit_b_32(num_classes=args_opt.num_classes, image_size=args_opt.resize, pretrained=args_opt.pretrained)
    elif args_opt.model == "vit_b_16":
        network = vit_b_16(num_classes=args_opt.num_classes, image_size=args_opt.resize, pretrained=args_opt.pretrained)
    elif args_opt.model == "vit_l_16":
        network = vit_l_16(num_classes=args_opt.num_classes, image_size=args_opt.resize, pretrained=args_opt.pretrained)
    elif args_opt.model == "vit_l_32":
        network = vit_l_32(num_classes=args_opt.num_classes, image_size=args_opt.resize, pretrained=args_opt.pretrained)

    # Define loss function.
    network_loss = CrossEntropySmooth(sparse=True,
                                      reduction="mean",
                                      smooth_factor=args_opt.smooth_factor,
                                      classes_num=args_opt.num_classes)

    # Define eval metrics.
    eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                    'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}

    # Init the model.
    model = Model(network, network_loss, metrics=eval_metrics)

    # Begin to eval.
    result = model.eval(dataset_eval)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ViT eval.")
    parser.add_argument("--model", required=False, default="vit_b_32",
                        choices=["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"])
    parser.add_argument("--device_target", type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument("--data_url", required=True, default=None, help="Location of data.")
    parser.add_argument("--pretrained", type=bool, default=False, help="Load pretrained model.")
    parser.add_argument("--num_parallel_workers", type=int, default=8, help="Number of parallel workers.")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of batch size.")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of classification.")
    parser.add_argument("--smooth_factor", type=float, default=0.1, help="The smooth factor.")
    parser.add_argument("--resize", type=int, default=224, help="Resize the image.")

    args = parser.parse_known_args()[0]
    vit_eval(args)
