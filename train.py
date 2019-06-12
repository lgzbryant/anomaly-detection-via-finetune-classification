#_*_coding:utf-8_*_

#author: lgz

#date: 19-6-6

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import os
from trainer import train_model
from visualization import visualize_model

from losses.cos_face_loss import  CosineMarginProduct
from losses.arc_face_loss import  ArcMarginProduct
from losses.linear_loss import InnerProduct

from config import TrainOptions
args = TrainOptions().parse()

from datasets import CreateDataloader
dataloaders, dataset_sizes, class_names = CreateDataloader(args)


save_model = args.save_model
if not os.path.exists(save_model):
    os.mkdir(save_model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


##########################################
from backbones.resnet import resnet18
from backbones.densenet import densenet169


if args.backbone == 'resnet18':
    model_ft = resnet18()
    pretrained_dict = models.resnet18(pretrained=True).state_dict()
    model_output_dimension = 512

elif args.backbone == 'densenet169':
    model_ft = densenet169()
    pretrained_dict = models.densenet169(pretrained=True).state_dict()
    model_output_dimension = 1664

else:
    print(args.backbone, ' is not available!')


# print(model_ft)
###########################################

model_dict = model_ft.state_dict()

pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

model_dict.update(pretrained_dict)

model_ft.load_state_dict(model_dict)

if args.finetune_last_layer == True:

    for param in model_ft.parameters():
        param.requires_grad = False

##########################################
if args.loss_type == 'ArcFace':
    margin = ArcMarginProduct(model_output_dimension, 2)
elif args.loss_type == 'CosFace':
    margin = CosineMarginProduct(model_output_dimension, 2)
elif args.loss_type == 'SphereFace':
    pass
elif args.loss_type == 'Softmax':
    margin = InnerProduct(model_output_dimension, 2)
else:
    print(args.loss_type, 'is not available!')

#################################################

model_ft = model_ft.to(device)
margin = margin.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=0.000001, nesterov=True)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=49, gamma=0.1)


model_ft = train_model(dataloaders, dataset_sizes,  model_ft, criterion, args.save_loss_dir, margin,
                       optimizer_ft, exp_lr_scheduler, num_epochs=args.num_epoch)

torch.save(model_ft.state_dict(), save_model+'/' + 'finetune.pkl')

visualize_model(model_ft, dataloaders, class_names, margin)

