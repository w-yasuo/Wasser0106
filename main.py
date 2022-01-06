import os
import torch
import argparse
from train import train
from data import SelfDatasetFolder
from model import Wasserstein_SqueezeNet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', metavar='DIR', default=r'D:\Datasets\1228_same',
                    help='path to dataset')
parser.add_argument('-save', metavar='DIR', default=r'D:\Checkpoints\Wasser_0105',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='SqueezeNet',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=48, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default=r'', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_loss = None


def main():
    global best_loss
    args = parser.parse_args()
    model = Wasserstein_SqueezeNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if os.path.exists(args.resume):
        print(" Load {}".format(args.resume))
        model = Wasserstein_SqueezeNet()
        model_CKPT = torch.load(args.resume)
        args.start_epoch = model_CKPT['epoch']
        best_loss = model_CKPT['best_loss']
        model.load_state_dict(model_CKPT['state_dict'])
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(model_CKPT['optimizer'])
        lr = (param_group['lr'] for param_group in optimizer.param_groups)
        print("start_epoch:{} best_loss:{} LR:{}".format(args.start_epoch, best_loss, lr))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    device = args.gup if torch.cuda.is_available() else 'cpu'
    print(" Useing the {} to trining".format(device))
    model.to(device)
    train_data_path = os.path.join(args.data, "train")
    val_data_path = os.path.join(args.data, "val")
    train_dataset = SelfDatasetFolder(
        train_data_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    test_val_loader = torch.utils.data.DataLoader(
        SelfDatasetFolder(val_data_path),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # train for one epoch
    train(args, train_loader, test_val_loader, model, optimizer, device, best_loss)


if __name__ == '__main__':
    main()
