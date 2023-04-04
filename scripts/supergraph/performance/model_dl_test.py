# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch_testbed import cifar10_models
from torch_testbed.dataloader_dali import cifar10_dataloaders
from torch_testbed.timing import MeasureTime, clear_timings, print_all_timings
from archai.common import utils

utils.setup_cuda(42, local_rank=0)

batch_size = 512
half = True

datadir = utils.full_path("~/dataroot")
train_dl, test_dl = cifar10_dataloaders(datadir, train_batch_size=batch_size, test_batch_size=1024, cutout=0)

model = cifar10_models.resnet18().cuda()
lr, momentum, weight_decay = 0.025, 0.9, 3.0e-4
optim = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
crit = torch.nn.CrossEntropyLoss().cuda()

if half:
    model = model.half()
    crit = crit.half()


@MeasureTime
def iter_dl(dl):
    i, d = 0, 0
    for x, l in dl:
        x, l = x.cuda().half() if half else x.cuda(), l.cuda()
        y = model(x)
        loss = crit(y, l)
        optim.zero_grad()
        loss.backward()
        optim.step()
        i += 1
        d += len(x)
    return i, d


def warm_up(epochs):
    for _ in range(epochs):
        train_dl = [
            (
                torch.rand(batch_size, 3, 12, 12).cuda()
                if not half
                else torch.rand(batch_size, 3, 12, 12).cuda().half(),
                torch.LongTensor(batch_size).random_(0, 10).cuda(),
            )
            for _ in range(round(50000 / batch_size))
        ]
        i, d = iter_dl(train_dl)


# warm_up(5)
# cudnn.benchmark = False

print_all_timings()
clear_timings()

for _ in range(5):
    i, d = iter_dl(train_dl)

print_all_timings()
print(i, d)

exit(0)
