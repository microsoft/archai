import argparse
import math
import os
import time
from typing import List, Mapping, Optional, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import yaml
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from archai.common.ordered_dict_logger import get_global_logger
from archai.supergraph.algos.nasbench101.nasbench101_dataset import Nasbench101Dataset
from archai.common import common, utils

logger = get_global_logger()


def train(
    epochs, train_dl, val_dal, net, device, crit, optim, sched, sched_on_epoch, half, quiet, grad_clip: float
) -> List[Mapping]:
    train_acc, _ = 0.0, 0.0
    metrics = []
    for epoch in range(epochs):
        lr = optim.param_groups[0]["lr"]
        train_acc, loss = train_epoch(epoch, net, train_dl, device, crit, optim, sched, sched_on_epoch, half, grad_clip)

        val_acc = test(net, val_dal, device, half) if val_dal is not None else math.nan
        metrics.append({"val_top1": val_acc, "train_top1": train_acc, "lr": lr, "epoch": epoch, "train_loss": loss})
        if not quiet:
            logger.info(f"train_epoch={epoch}, val_top1={val_acc}," f" train_top1={train_acc}, lr={lr:.4g}")
    return metrics


def optim_sched_resnet(net, epochs):
    lr, momentum, weight_decay = 0.1, 0.9, 1.0e-4
    optim = torch.optim.SGD(net.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    logger.info(f"lr={lr}, momentum={momentum}, weight_decay={weight_decay}")

    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100, 150, 200, 400, 600])  # resnet original paper
    sched_on_epoch = True

    logger.info(f"sched_on_epoch={sched_on_epoch}, sched={str(sched)}")

    return optim, sched, sched_on_epoch


def optim_sched_paper(net, epochs):
    lr, momentum, weight_decay = 0.2, 0.9, 0.0001
    optim = torch.optim.RMSprop(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    logger.info(f"optim=RMSprop, lr={lr}, momentum={momentum}, weight_decay={weight_decay}")

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    sched_on_epoch = True

    logger.info(f"sched_on_epoch={sched_on_epoch}, sched={str(sched)}")

    return optim, sched, sched_on_epoch


def optim_sched_darts(net, epochs):
    lr, momentum, weight_decay = 0.025, 0.9, 3.0e-4
    optim = torch.optim.SGD(net.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    logger.info(f"optim=SGD, lr={lr}, momentum={momentum}, weight_decay={weight_decay}")

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    sched_on_epoch = True

    logger.info(f"sched_on_epoch={sched_on_epoch}, sched={str(sched)}")

    return optim, sched, sched_on_epoch


def get_data(
    datadir: str,
    train_batch_size=256,
    test_batch_size=256,
    cutout=0,
    train_num_workers=-1,
    test_num_workers=-1,
    val_percent=20.0,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    if utils.is_debugging():
        train_num_workers = test_num_workers = 0
        logger.info("debugger=true, num_workers=0")
    if train_num_workers <= -1:
        train_num_workers = torch.cuda.device_count() * 4
    if test_num_workers <= -1:
        test_num_workers = torch.cuda.device_count() * 4

    train_transform = cifar10_transform(aug=True, cutout=cutout)
    trainset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True, transform=train_transform)

    val_len = int(len(trainset) * val_percent / 100.0)
    train_len = len(trainset) - val_len

    valset = None
    if val_len:
        trainset, valset = torch.utils.data.random_split(trainset, [train_len, val_len])

    train_dl = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=train_num_workers, pin_memory=True
    )

    if valset is not None:
        val_dl = torch.utils.data.DataLoader(
            valset, batch_size=test_batch_size, shuffle=False, num_workers=test_num_workers, pin_memory=True
        )
    else:
        val_dl = None

    test_transform = cifar10_transform(aug=False, cutout=0)
    testset = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True, transform=test_transform)
    test_dl = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=test_num_workers, pin_memory=True
    )

    logger.info(f"train_len={train_len}, val_len={val_len}, test_len={len(testset)}")

    return train_dl, val_dl, test_dl


def train_epoch(
    epoch, net, train_dl, device, crit, optim, sched, sched_on_epoch, half, grad_clip: float
) -> Tuple[float, float]:
    correct, total, loss_total = 0, 0, 0.0
    net.train()
    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if half:
            inputs = inputs.half()

        outputs, loss = train_step(net, crit, optim, sched, sched_on_epoch, inputs, targets, grad_clip)
        loss_total += loss

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    if sched and sched_on_epoch:
        sched.step()
    return 100.0 * correct / total, loss_total


def train_step(
    net: nn.Module,
    crit: _Loss,
    optim: Optimizer,
    sched: _LRScheduler,
    sched_on_epoch: bool,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    grad_clip: float,
) -> Tuple[torch.Tensor, float]:
    outputs = net(inputs)

    loss = crit(outputs, targets)
    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), grad_clip)

    optim.step()
    if sched and not sched_on_epoch:
        sched.step()
    return outputs, loss.item()


def test(net, test_dl, device, half) -> float:
    correct, total = 0, 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs = inputs.to(device, non_blocking=False)
            targets = targets.to(device)

            if half:
                inputs = inputs.half()

            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def param_size(model: torch.nn.Module) -> int:
    """count all parameters excluding auxiliary"""
    return sum(v.numel() for name, v in model.named_parameters() if "auxiliary" not in name)


def cifar10_transform(aug: bool, cutout=0):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]

    transf = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]

    if aug:
        aug_transf = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        transf = aug_transf + transf

    if cutout > 0:  # must be after normalization
        transf += [CutoutDefault(cutout)]

    return transforms.Compose(transf)


class CutoutDefault:
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def log_metrics(
    expdir: str, filename: str, metrics, test_acc: float, args, nsds: Nasbench101Dataset, model_id: int
) -> None:
    print(
        f"filename: {filename}",
        f"test_acc: {test_acc}",
        f"nasbenc101_test_acc: {nsds.get_test_acc(model_id)}",
        metrics[-1],
    )
    results = [
        ("test_acc", test_acc),
        ("nasbenc101_test_acc", nsds.get_test_acc(model_id)),
        ("val_acc", metrics[-1]["val_top1"]),
        ("epochs", args.epochs),
        ("train_batch_size", args.train_batch_size),
        ("test_batch_size", args.test_batch_size),
        ("model_name", args.model_name),
        ("exp_name", args.experiment_name),
        ("exp_desc", args.experiment_description),
        ("seed", args.seed),
        ("devices", utils.cuda_device_names()),
        ("half", args.half),
        ("cutout", args.cutout),
        ("train_acc", metrics[-1]["train_top1"]),
        ("loader_workers", args.loader_workers),
        ("date", str(time.time())),
    ]
    utils.append_csv_file(os.path.join(expdir, f"{filename}.tsv"), results)
    with open(os.path.join(expdir, f"{filename}_metrics.yaml"), "w") as f:
        yaml.dump(metrics, f)
    with open(os.path.join(expdir, f"{filename}_nasbench101.yaml"), "w") as f:
        yaml.dump(nsds[model_id], f)


def create_crit(device, half):
    crit = nn.CrossEntropyLoss().to(device)
    if half:
        crit.half()
    return crit


def create_model(nsds, index, device, half) -> nn.Module:
    net = nsds.create_model(index)
    logger.info(f"param_size_m={param_size(net):.1e}")
    net = net.to(device)
    if half:
        net.half()
    return net


def main():
    parser = argparse.ArgumentParser(description="Pytorch cifar training")
    parser.add_argument("--experiment-name", "-n", default="train_pytorch")
    parser.add_argument("--experiment-description", "-d", default="Train cifar usin pure PyTorch code")
    parser.add_argument("--epochs", "-e", type=int, default=108)
    parser.add_argument("--model-name", "-m", default="5")
    parser.add_argument("--device", default="", help='"cuda" or "cpu" or "" in which case use cuda if available')
    parser.add_argument("--train-batch-size", "-b", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--seed", "-s", type=float, default=42)
    parser.add_argument("--half", type=lambda x: x.lower() == "true", nargs="?", const=True, default=False)
    parser.add_argument("--cutout", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--datadir", default="", help="where to find dataset files, default is ~/torchvision_data_dir")
    parser.add_argument("--outdir", default="", help="where to put results, default is ~/logdir")

    parser.add_argument(
        "--loader-workers", type=int, default=-1, help="number of thread/workers for data loader (-1 means auto)"
    )

    args = parser.parse_args()

    if not args.datadir:
        args.datadir = common.default_dataroot()
    nsds_dir = args.datadir
    if os.environ.get("PT_DATA_DIR", ""):
        nsds_dir = os.environ.get("PT_DATA_DIR")
    if not args.outdir:
        args.outdir = os.environ.get("PT_OUTPUT_DIR", "")
        if not args.outdir:
            args.outdir = os.path.join("~/logdir", "nasbench101", args.experiment_name)
    assert isinstance(nsds_dir, str)

    expdir = utils.full_path(args.outdir)
    os.makedirs(expdir, exist_ok=True)

    utils.setup_cuda(args.seed)
    datadir = utils.full_path(args.datadir)
    os.makedirs(datadir, exist_ok=True)

    # log config for reference
    logger.info(f'exp_name="{args.experiment_name}", exp_desc="{args.experiment_description}"')
    logger.info(f'model_name="{args.model_name}", seed={args.seed}, epochs={args.epochs}')
    logger.info(f"half={args.half}, cutout={args.cutout}")
    logger.info(f'datadir="{datadir}"')
    logger.info(f'expdir="{expdir}"')
    logger.info(f"train_batch_size={args.train_batch_size}")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nsds = Nasbench101Dataset(os.path.join(nsds_dir, "nasbench_ds", "nasbench_full.pkl"))

    # load data just before train start so any errors so far is not delayed
    train_dl, val_dl, test_dl = get_data(
        datadir=datadir,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        train_num_workers=args.loader_workers,
        test_num_workers=args.loader_workers,
        cutout=args.cutout,
    )

    model_id = int(args.model_name)  # 5, 401, 4001, 40001, 400001
    epochs = args.epochs

    net = create_model(nsds, model_id, device, args.half)
    crit = create_crit(device, args.half)
    optim, sched, sched_on_epoch = optim_sched_darts(net, epochs)  # optim_sched_darts optim_sched_paper

    train_metrics = train(
        epochs,
        train_dl,
        val_dl,
        net,
        device,
        crit,
        optim,
        sched,
        sched_on_epoch,
        args.half,
        False,
        grad_clip=args.grad_clip,
    )
    test_acc = test(net, test_dl, device, args.half)
    log_metrics(expdir, f"metrics_{model_id}", train_metrics, test_acc, args, nsds, model_id)


if __name__ == "__main__":
    main()
