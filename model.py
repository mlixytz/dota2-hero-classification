import argparse
import os
import shutil
import torch
import torch.nn as nn
import time

from torch.utils.tensorboard import SummaryWriter


from efficientnet_pytorch.model import EfficientNet
from dataset import build_train_dataset, build_val_dataset


class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} (avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProcessMeter:
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best, checkpoint_dir, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, 'best.pth.tar'))


def train(dataloader, model, criterion, optimizer, epoch, writer):
    model.train()

    for i, (images, target) in enumerate(dataloader):
        output = model(images)
        loss = criterion(output, target)
        print('Epoch ', epoch, loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss, i)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(dataloader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProcessMeter(len(dataloader), batch_time, losses, top1, top5,
                            prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.perf_counter()
        for i, (images, target) in enumerate(dataloader):
            images = images.cuda()
            target = target.cuda()

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, (1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            if i % args.print_freq == 0:
                progress.print(i)

    return top1.avg


def main():
    parser = argparse.ArgumentParser(description="dota2 hero classification")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--image_size', default=224, type=int,
                        help='image size')
    parser.add_argument('--classes_num', type=int, default=123)
    parser.add_argument('--advprop', default=False, action='store_true',
                        help='use advprop or not')

    parser.add_argument('--train_data', type=str, help='path to train dataset')
    parser.add_argument('--val_data', type=str, help='path to val dataset')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--summary_dir', type=str, default='./runs')
    parser.add_argument('--print_freq', type=int, default=10)

    args = parser.parse_args()

    best_acc1 = 0

    # 构建模型
    if args.pretrained:
        model = EfficientNet.from_pretrained(args.arch,
                                             num_classes=args.classes_num,
                                             advprop=args.advprop)
        print(f"=> using pre-trained model '{args.arch}'")
    else:
        print(f"=> creating model '{args.arch}'")
        model = EfficientNet.from_name(args.arch, override_params={
            'num_classes': args.classes_num})
    model.cuda()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optimizer.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.wd)

    # 准备数据
    train_dataset = build_train_dataset(args.image_size, args.train_data)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=True)

    val_dataset = build_val_dataset(args.image_size, args.val_data)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=True)

    writer = SummaryWriter(args.summary_dir)
    # 迭代
    for epoch in range(args.epochs):
        train(train_dataloader, model, criterion, optimizer, epoch, writer)

        # 评估
        acc1 = validate(val_dataloader, model, criterion, args)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)

        save_checkpoint(model.state_dict(), is_best,
                        os.path.join(f'checkpoint.pth.tar.epoch_{epoch}'))


if __name__ == "__main__":
    main()
