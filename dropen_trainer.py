import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
import torch.autograd as autograd
import sys
import os
import gc
import json

from utils.Empirical.utils_ensemble import AverageMeter, accuracy, test, copy_code, requires_grad_
from models.ensemble import Ensemble
from utils.Empirical.utils_ensemble import Cosine, Magnitude
from utils.Empirical.third_party.distillation import Linf_distillation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def PGD(models, inputs, labels, eps):
    steps = 6
    alpha = eps / 3.

    #print(inputs.shape)
    adv = inputs.detach() + torch.FloatTensor(inputs.shape).uniform_(-eps, eps).to(device)
    adv = torch.clamp(adv, 0, 1)
    criterion = nn.CrossEntropyLoss()

    adv.requires_grad = True
    for _ in range(steps):
        #adv.requires_grad_()
        grad_loss = 0
        for i, m in enumerate(models):
            
            loss = criterion(m(adv), labels)
            grad = autograd.grad(loss, adv, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grad_loss += Magnitude(grad)

        #grad_loss /= 3
        grad_loss.backward()
        sign_grad = adv.grad.data.sign()
        with torch.no_grad():
            adv.data = adv.data + alpha * sign_grad
            adv.data = torch.max(torch.min(adv.data, inputs + eps), inputs - eps)
            adv.data = torch.clamp(adv.data, 0., 1.)

    adv.grad = None
    return adv.detach()

def TRS_Trainer(args, loader: DataLoader, models, criterion, optimizer: Optimizer,
                epoch: int, device: torch.device, writer=None):

    with torch.autograd.set_detect_anomaly(True):
        for i in range(1):
            #Set the mode to training
            models[i].train()
            requires_grad_(models[i], True)

        #initiate gradient memory
        gradient_memory = []
        
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time

            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            inputs.requires_grad = True
            grads = []
            #The standard loss of model training
            #print("Calculate standard loss")
            loss_std = 0
            for j in range(1):
                logits = models[j](inputs)
                loss = criterion(logits, targets)
                grad = autograd.grad(loss, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                grads.append(grad)
                loss_std += loss

            cos_loss, smooth_loss = 0, 0


            gradient_memory.append(grads[0].detach())
            #print(grads[0].shape)
            #print(gradient_memory)
            if len(gradient_memory) > 1:
                #print("Memory full")
                for i in range(0,len(gradient_memory)-1):
                    cos_loss = cos_loss + Cosine(grads[0],gradient_memory[i])

                cos_loss = cos_loss / float(len(gradient_memory)-1)
            
            if len(gradient_memory) > 5:
                with torch.no_grad():
                    print("Length of gradient memorized: {}".format(len(gradient_memory)))
                    grad_to_desert = gradient_memory.pop(0)
                    grad_to_desert.detach()
                    grad_to_desert = None
                    del grad_to_desert     
                    torch.cuda.empty_cache()
                    gc.collect()
            #print("Success in calculating cos loss")
            
            #Vanilla PGD training
            N = inputs.shape[0] // 2
            #The current radius of ball
            cureps = (args.adv_eps - args.init_eps) * epoch / args.epochs + args.init_eps
            clean_inputs = inputs[:N].detach() 
            #Generate pgd pertubed adv examples
            #print("Calculate PGD pertubation examples")
            adv_inputs = PGD(models, inputs[N:].detach(), targets[N:].detach(), cureps).detach()
            #print("Success in calculating PGD example")
            adv_x = torch.cat([clean_inputs, adv_inputs])

            adv_x.requires_grad = True

            if (args.plus_adv):
                for j in range(1):
                    outputs = models[j](adv_x)
                    loss = criterion(outputs, targets)
                    grad = autograd.grad(loss, adv_x, create_graph=True)[0]
                    grad = grad.flatten(start_dim=1)
                    smooth_loss += Magnitude(grad)

            else:
                # grads = []
                for j in range(1):
                    outputs = models[j](inputs)
                    loss = criterion(outputs, targets)
                    grad = autograd.grad(loss, inputs, create_graph=True)[0]
                    grad = grad.flatten(start_dim=1)
                    smooth_loss += Magnitude(grad)

            #smooth_loss /= 3

            #print(smooth_loss.shape)
            loss = loss_std + args.scale * (args.coeff * cos_loss+ args.lamda * smooth_loss * 1e4)



            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits.detach(), targets.detach(), topk=(1, 5))
            print("acc1：{},acc5:{}".format(acc1,acc5))
            print("loss_std:{},cos_loss{},smooth_loss{}".format(loss_std,cos_loss,smooth_loss))

            #print("Backward integrated loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with open('commandline_args.txt', 'w') as f:
                json.dump(args.__dict__, f, indent=2)
            
            for i in range(1):
                model_path_i = "./dropen_model/TRS_{}.pth.tar".format(epoch)
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': models[i].state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, model_path_i)


def Naive_adv_trainer(args, loader: DataLoader, models, criterion, optimizer: Optimizer,
                epoch: int, device: torch.device, writer=None):
    with torch.autograd.set_detect_anomaly(True):
        for i in range(1):
            #Set the mode to training
            models[i].train()
            requires_grad_(models[i], True)
        
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time

            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            inputs.requires_grad = True
            grads = []
            #The standard loss of model training
            #print("Calculate standard loss")



            N = inputs.shape[0] // 2
            #The current radius of ball
            cureps = (args.adv_eps - args.init_eps) * epoch / args.epochs + args.init_eps
            clean_inputs = inputs[:N].detach()    # PGD(self.models, inputs[:N], targets[:N])
            #Generate pgd perturbed adv examples
            #print("Calculate PGD pertubation examples")
            adv_inputs = PGD(models, inputs[N:], targets[N:], cureps).detach()
            #print("Success in calculating PGD example")
            adv_x = torch.cat([clean_inputs, adv_inputs])

            adv_x.requires_grad = True

            loss = 0
            for j in range(1):
                outputs = models[j](adv_x)
                loss = loss + criterion(outputs, targets)



            # measure accuracy and record loss
            logits = models[0](inputs)
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            print("acc1：{},acc5:{}".format(acc1,acc5))
            #print("loss_std:{},cos_loss:{},smooth_loss:{}".format(loss_std,cos_loss,smooth_loss))

            #print("Backward integrated loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with open('commandline_args.txt', 'w') as f:
                json.dump(args.__dict__, f, indent=2)

            for i in range(1):
                model_path_i = "./dropen_model/naive_{}.pth.tar".format(epoch)
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': models[i].state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, model_path_i)

def Naive_normal_trainer(args, loader: DataLoader, models, criterion, optimizer: Optimizer,
                epoch: int, device: torch.device, writer=None):
    
    with torch.autograd.set_detect_anomaly(True):
        for i in range(1):
            #Set the mode to training
            models[i].train()
            requires_grad_(models[i], True)
        
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time

            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True
            #The standard loss of model training
            #print("Calculate standard loss")

            loss = 0
            for j in range(1):
                outputs = models[j](inputs)
                loss = loss + criterion(outputs, targets)



            # measure accuracy and record loss
            logits = models[0](inputs)
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            print("acc1：{},acc5:{}".format(acc1,acc5))
            #print("loss_std:{},cos_loss:{},smooth_loss:{}".format(loss_std,cos_loss,smooth_loss))

            #print("Backward integrated loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with open('commandline_args.txt', 'w') as f:
                json.dump(args.__dict__, f, indent=2)
            
            for i in range(1):
                model_path_i = "./dropen_model/pure_{}.pth.tar".format(epoch)
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': models[i].state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, model_path_i)
