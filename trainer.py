#_*_coding:utf-8_*_

#author: lgz

#date: 19-6-6


from __future__ import print_function, division
import torch
import numpy as np
import time
import os
import copy
from sklearn import metrics
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(dataloaders, dataset_sizes, model, criterion, save_loss_path,
                            margin, optimizer, scheduler, num_epochs=25):

    file_name = os.path.join(save_loss_path, 'loss.txt')
    # print(file_name)
    with open(file_name, 'w+') as file:
        file.write('====================LOSS=====================')

    show_train_acc = []
    show_train_auc = []
    show_val_acc = []
    show_val_auc = []
    show_epoch = []
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_auc = 0.0

    for epoch in range(num_epochs):
        show_epoch.append(epoch)

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        with open(file_name, 'a+') as file:
            file.write('\n')
            file.write('\n')
            file.write('Epoch {}/{}'.format(epoch, num_epochs - 1))
            # file.write('\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
                train_scores_auc = []
                train_labels_auc = []
            else:
                model.eval()   # Set model to evaluate mode
                val_scores_auc = []
                val_labels_auc = []

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    # print(outputs.size())
                    # print(labels.size())
                    # margin is for chosing: Linear, arc_face_loss,etc
                    outputs = margin(outputs, labels)

                    loss = criterion(outputs, labels)

                    sout, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                labels_cpu = (labels.data.cpu().numpy()).tolist()
                for each_label in labels_cpu:
                    if phase=='train':
                        train_labels_auc.append(each_label)
                    else :
                        val_labels_auc.append(each_label)

                scores_cpu = (sout.detach().data.cpu().numpy()).tolist()
                for each_score in scores_cpu:
                    if phase=='train':
                        train_scores_auc.append(each_score)
                    else :
                        val_scores_auc.append(each_score)


            if phase == 'train':
                scores_auc = train_scores_auc
                labels_auc = train_labels_auc
            else:
                scores_auc = val_scores_auc
                labels_auc = val_labels_auc

            scores_auc = np.array(scores_auc)
            labels_auc = np.array(labels_auc)

            fpr, tpr, thresholds = metrics.roc_curve(labels_auc, scores_auc)
            # print('fpr: ', fpr, '\n','tpr: ', tpr, '\n', 'th :', thresholds)
            # print(metrics.auc(fpr, tpr))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            AUC = metrics.auc(fpr, tpr)

            if phase == 'train':
                show_train_auc.append(AUC)
                show_train_acc.append(epoch_acc.item())
            else:
                show_val_acc.append(epoch_acc.item())
                show_val_auc.append(AUC)

            print('{} Loss: {:.4f}   Acc: {:.4f}   AUC: {:.4f}'.format(phase, epoch_loss,
                                                                   epoch_acc, AUC ))

            # file_name = os.path.join(save_loss_path, 'loss.txt')
            with open(file_name, 'a+') as loss_file:
                loss_file.write('\n')

                loss_file.write('{} Loss: {:.4f}   Acc: {:.4f}   AUC: {:.4f}'.format(phase, epoch_loss,
                                                                   epoch_acc, AUC))
                # loss_file.write('===================================================================\n')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val' and AUC > best_auc:
                best_auc = AUC




        print()

    plt.figure(figsize=(10, 5))
    plt.title("ACC and AUC During Training and validation")
    plt.plot(show_epoch,  show_train_acc, label="train_acccuracy")
    plt.plot(show_epoch, show_train_auc, label="train_AUC")
    plt.plot(show_epoch, show_val_acc, label="val_acccuracy")
    plt.plot(show_epoch, show_val_auc, label="val_AUC")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend(loc='lower right', fontsize=16)
    plt.savefig('show_loss_value/ACC and AUC.png')
    plt.show()


    time_elapsed = time.time() - since
    print('Cost time: {:.0f}hours ({:.0f}minutes {:.0f}s)'.format(time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val AUC: {:4f}'.format(best_auc))

    with open(file_name, 'a+') as loss_file:
        loss_file.write('\n')
        loss_file.write('\n')
        loss_file.write('Cost time {:.0f}hours ({:.0f}minutes {:.0f}s)'.format(time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))
        loss_file.write('\n')
        loss_file.write('Best val Acc: {:4f}'.format(best_acc))
        loss_file.write('\n')
        loss_file.write('Best val AUC: {:4f}'.format(best_auc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model