import modules
import time
import torch as th
import torchvision as tv
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from typing import Optional
import utils
import os
import yaml


def main(rank: int,
         world_size: int,
         batch_size: int,
         epochs: int,
         load_path: Optional[str],
         yaml_path: str,
         save_path: str,
         data_path: str,
         num_workers: int,
         port: str,
         save_every: int) -> None:

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    th.cuda.set_device(rank)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    transform = []
    if config["horizontal_flip"]:
        transform += [tv.transforms.RandomHorizontalFlip()]
    transform += [
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5], std=[0.5])
    ]

    transform = tv.transforms.Compose(transform)
    dataset = tv.datasets.ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=True,
                            sampler=DistributedSampler(dataset),
                            num_workers=num_workers)
    if rank == 0:
        print(f"{len(dataset)} images in {data_path}, {len(dataloader)} batches of size {batch_size}, {world_size} gpus.")  # noqa: E501
    model = modules.Diffuser(in_channels=config["num_channels"],
                             out_channels=config["num_channels"],
                             T=config["diffuser"]["T"],
                             linear=config["diffuser"]["linear"],
                             unet_cfg=config["diffuser"]["unet"]).to(rank)
    if rank == 0:
        ema = th.optim.swa_utils.AveragedModel(model,
                                               multi_avg_fn=th.optim.swa_utils.get_ema_multi_avg_fn(
                                                   config["ema_decay"]))
        print(f"Loaded {yaml_path} model with {utils.count_parameters(model)} parameters.")
    else:
        ema = None
    model = DDP(model, device_ids=[rank])
    optim = th.optim.Adam(model.parameters(), lr=config["learning_rate"])
    if load_path is not None:
        utils.load_state(checkpoint_path=load_path,
                         model=model.module,
                         ema=ema,
                         optimizer=optim)
        if rank == 0:
            print(f"Resuming from {load_path}.")

    dist.barrier()
    if rank == 0:
        print(f"Starting training, {epochs} epochs.", flush=True)
    for epoch in range(1, epochs + 1):
        dataloader.sampler.set_epoch(epoch)
        if rank == 0:
            avg_loss = 0
            th.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
        for x, _ in dataloader:
            x = x.to(rank)
            optim.zero_grad()

            loss = model(x)

            if not th.isfinite(loss):
                raise RuntimeError("Loss is not finite.")

            loss.backward()
            optim.step()
            if rank == 0:
                ema.update_parameters(model)
                avg_loss += loss.item() / len(dataloader)
        dist.barrier()
        if rank == 0:
            if epoch % save_every == 0:
                utils.save_state(checkpoint_path=os.path.join(save_path, str(epoch * len(dataloader)).zfill(8) + ".pt"),
                                 model=model.module,
                                 ema=ema,
                                 optimizer=optim)
            print(f"Epoch {str(epoch).zfill(len(str(epochs)))}/{epochs}, Avg Loss: {avg_loss:.6e}, \
                    Time: {time.perf_counter() - start:.2f} s, Max Mem: {th.cuda.max_memory_allocated() / 1e9:.2f} GB",
                  flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=1, help="batch size (per gpu)")
    parser.add_argument("--nw", type=int, default=0, help="number of workers (per gpu)")
    parser.add_argument("--e", type=int, default=1, help="epochs")
    parser.add_argument("--p2l", type=str, default=None, help="path to load")
    parser.add_argument("--p2y", type=str, required=True, help="path to yaml")
    parser.add_argument("--p2s", type=str, required=True, help="path to save")
    parser.add_argument("--p2d", type=str, required=True, help="path to data")
    parser.add_argument("--port", type=str, default="12355", help="port")
    parser.add_argument("--se", type=int, default=1, help="save every n epochs")
    args = parser.parse_args()

    world_size = th.cuda.device_count()
    mp.spawn(main,
             args=(world_size,
                   args.bs,
                   args.e,
                   args.p2l,
                   args.p2y,
                   args.p2s,
                   args.p2d,
                   args.nw,
                   args.port,
                   args.se),
             nprocs=world_size)
