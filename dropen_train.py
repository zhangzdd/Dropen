import torch
import torchvision
from torch.utils.data import DataLoader
import argparse
from dropen_models import *
import os
import torch.optim as optim
from dropen_trainer import *
import torchvision.transforms as transforms
import gc

batch_size_train = 1000
batch_size_test = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=50,type=int, metavar='N',
                    help='batchsize (default: 50')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=40,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--num-models', type=int,default=1)

parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from existing checkpoint')
#parser.add_argument('--adv-training', action='store_true')
parser.add_argument('--adv-training', default=True)
parser.add_argument('--epsilon', default=512, type=float)
parser.add_argument('--num-steps', default=4, type=int)

# DRT Training params
parser.add_argument('--coeff', default=2.0, type=float)
parser.add_argument('--lamda', default=2.0, type=float)
parser.add_argument('--scale', default=5.0, type=float)
#parser.add_argument('--plus-adv', action='store_true')
parser.add_argument('--plus-adv', default=True)
parser.add_argument('--adv-eps', default=0.2, type=float)
parser.add_argument('--init-eps', default=0.1, type=float)

args = parser.parse_args()


args.epsilon /= 256.0



def train_mnist():
    train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_test, shuffle=True)

    model = [LeNet_with_dropout()]

    criterion = nn.CrossEntropyLoss().cuda()

    param = model[0].parameters()

    # optimizer = optim.SGD(param, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(param, lr=args.lr, weight_decay=args.weight_decay, eps=1e-7)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    #model_path = os.path.join(args.outdir, 'checkpoint.pth.tar')

    for epoch in range(args.epochs):

        TRS_Trainer(args, train_loader, model, criterion, optimizer, epoch, device)
        #test(test_loader, model, criterion, epoch, device)
        #Naive_normal_trainer(args, train_loader, model, criterion, optimizer, epoch, device)
        
        scheduler.step(epoch)


def train_cifar():
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.CIFAR10('./data_cifar/', train=True, download=True,
                             transform=transforms.Compose([
                                transforms.Pad(4),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32),
                                transforms.ToTensor()])),
                              batch_size=batch_size_train, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('./data_cifar/', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.Pad(4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32),
                                    transforms.ToTensor()])),
                              batch_size=batch_size_test, shuffle=True)

    model = [conv_dropen().to(device)]

    criterion = nn.CrossEntropyLoss()

    param = model[0].parameters()

    optimizer = optim.Adam(param, lr=args.lr, weight_decay=args.weight_decay, eps=1e-7)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    #model_path = os.path.join(args.outdir, 'checkpoint.pth.tar')

    for epoch in range(args.epochs):

        #TRS_Trainer(args, train_loader, model, criterion, optimizer, epoch, device)
        #Naive_adv_trainer(args, train_loader, model, criterion, optimizer, epoch, device)
        Naive_normal_trainer(args, train_loader, model, criterion, optimizer, epoch, device)
        
        scheduler.step(epoch)




if __name__ == "__main__":
    #train_mnist()
    train_cifar()