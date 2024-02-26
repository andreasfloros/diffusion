import math
import modules
import os
import torch as th
import torchvision as tv
from typing import Optional
import utils
import uuid
import yaml


@th.inference_mode()
def main(checkpoint_path: str,
         yaml_path: str,
         out_path: str,
         num_samples: int,
         batch_size: int,
         use_ema: bool,
         eta: float,
         steps: Optional[int]) -> None:
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    model = modules.Diffuser(in_channels=config["num_channels"],
                             out_channels=config["num_channels"],
                             T=config["diffuser"]["T"],
                             linear=config["diffuser"]["linear"],
                             unet_cfg=config["diffuser"]["unet"]).eval().to(device)
    if use_ema:
        ema = th.optim.swa_utils.AveragedModel(model,
                                               multi_avg_fn=th.optim.swa_utils.get_ema_multi_avg_fn(
                                                   config["ema_decay"]))
        utils.load_state(checkpoint_path=checkpoint_path, ema=ema)
        model = ema.module
    else:
        utils.load_state(checkpoint_path=checkpoint_path, model=model)

    image_size = config["image_size"]
    scale = 2 ** len(model.nwds)
    coarse_shape = [image_size[0] // scale, image_size[1] // scale]
    num_batches = math.ceil(num_samples / batch_size)
    for _ in range(num_batches):
        imgs = th.randn(batch_size, config["num_channels"], *coarse_shape, device=device)
        imgs = model.sample(imgs, steps=steps, eta=eta)
        for img in imgs:
            tv.utils.save_image(img,
                                os.path.join(out_path, f"{uuid.uuid4()}.png"),
                                normalize=True,
                                value_range=(-1, 1))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, default=1, help="number of samples")
    parser.add_argument("--bs", type=int, default=1, help="batch size")
    parser.add_argument("--p2c", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--p2y", type=str, required=True, help="path to yaml")
    parser.add_argument("--p2o", type=str, required=True, help="path to output images")
    parser.add_argument("--ema", action="store_true", help="use ema")
    parser.add_argument("--eta", type=float, default=1., help="eta")
    parser.add_argument("--steps", type=int, default=None, help="steps")
    args = parser.parse_args()

    main(checkpoint_path=args.p2c,
         yaml_path=args.p2y,
         out_path=args.p2o,
         num_samples=args.ns,
         batch_size=args.bs,
         use_ema=args.ema,
         eta=args.eta,
         steps=args.steps)
