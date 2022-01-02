import argparse
import torch
from dropen_models import *
import json
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from dropen_trainer import *

parser = argparse.ArgumentParser(description='resume training for models')
parser.add_argument('--path', type=str, help='path of model to resume',default="./dropen_model/naive_22.pth.tar")
parser.add_argument('--type',type=str,help='type of dataset',default="CIFAR10")
#parser.add_argument('--epoch',type=int,help='epoch to start')
args = parser.parse_args()

print("Loading training params")
train_parser = argparse.ArgumentParser()
train_args = train_parser.parse_args()
with open('commandline_args.txt', 'r') as f:
    train_args.__dict__ = json.load(f)

model = conv_dropen()
checkpoint = torch.load(args.path)
model.load_state_dict(checkpoint["state_dict"])

if args.type == "CIFAR10":
    print("Resume CIFAR10 model training")
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.CIFAR10('./data_cifar/', train=True, download=True,
                             transform=transforms.Compose([
                                transforms.Pad(4),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32),
                                transforms.ToTensor()])),
                              batch_size=1000, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('./data_cifar/', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.Pad(4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32),
                                    transforms.ToTensor()])),
                              batch_size=1000, shuffle=True)

criterion = nn.CrossEntropyLoss().cuda()

param = model.parameters()

# optimizer = optim.SGD(param, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
print(checkpoint["optimizer"]["param_groups"])
#optimizer = optim.Adam(param).load_state_dict(checkpoint["optimizer"])
optimizer = optim.Adam(param, lr=checkpoint["optimizer"]["param_groups"][0]["lr"], weight_decay=checkpoint["optimizer"]["param_groups"][0]["weight_decay"], eps=checkpoint["optimizer"]["param_groups"][0]["eps"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

#model_path = os.path.join(args.outdir, 'checkpoint.pth.tar')


for epoch in range(checkpoint["epoch"],200):
    
    #TRS_Trainer(train_args, train_loader, [model.cuda()], criterion, optimizer, epoch, device)
    Naive_adv_trainer(train_args, train_loader, [model.cuda()], criterion, optimizer, epoch, device)
    #Naive_normal_trainer(train_args, train_loader, [model.cuda()], criterion, optimizer, epoch, device)
    scheduler.step(epoch)