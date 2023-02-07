# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch_testbed import cifar10_models, utils
from torch_testbed.timing import MeasureTime, print_all_timings


utils.setup_cuda(42, local_rank=0)

batch_size = 512
half = True
model = cifar10_models.resnet18().cuda()
lr, momentum, weight_decay = 0.025, 0.9, 3.0e-4
optim = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
crit = torch.nn.CrossEntropyLoss().cuda()

if half:
    model = model.half()
    crit = crit.half()


@MeasureTime
def iter_dl(ts):
    i, d = 0, 0
    for x, l in ts:
        y = model(x)
        loss = crit(y, l)
        optim.zero_grad()
        loss.backward()
        optim.step()
        i += 1
        d += len(x)
    return i, d


for _ in range(5):
    train_dl = [
        (
            torch.rand(batch_size, 3, 12, 12).cuda() if not half else torch.rand(batch_size, 3, 12, 12).cuda().half(),
            torch.LongTensor(batch_size).random_(0, 10).cuda(),
        )
        for _ in range(round(50000 / batch_size))
    ]
    i, d = iter_dl(train_dl)

print_all_timings()
print(i, d)

exit(0)
