from itertools import product
from itertools import combinations

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.generators import dataset_benchmark


import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter



import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_confusion_matrix
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import datasets

from itertools import product
from itertools import combinations


import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.animation import FuncAnimation ,FFMpegWriter ,PillowWriter

import dill
import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm
import copy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ListExperiences=list()

All=list(range(10))
for sample in range(2,9,1):
    S=combinations(range(10), sample)
    #All=list(range(10))
    for i in S :
        L1=list(i)
        L2=[k for k in All if k not in L1] 
        ListExperiences.append(L1)
len(ListExperiences)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class ClassSpecificImageFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root,
            dropped_classes=[],
            transform = None,
            target_transform = None,
            loader = datasets.folder.default_loader,
            is_valid_file = None,
    ):
        self.dropped_classes = dropped_classes
        super(ClassSpecificImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       is_valid_file=is_valid_file)
        self.imgs = self.samples

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = [c for c in classes if c not in self.dropped_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.0,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        # print("initialze model")
        for m in self.module_list:
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()


    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
                or isinstance(layer, ORU)
                or isinstance(layer, ERU)
            ):
                activations.append(x)
        return x, activations
def train(model, trainloader, optimizer, criterion,nb_classes):
    List_mx=[]
    model.train()
    #print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in enumerate(trainloader):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        mx=multiclass_confusion_matrix(preds ,labels,nb_classes,normalize="pred")
        List_mx.append(mx)
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc,List_mx


def validate(model, testloader, criterion,nb_classes):
    List_mx=[]
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
            mx=multiclass_confusion_matrix(preds ,labels,nb_classes,normalize="pred")
            List_mx.append(mx)
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc,List_mx
def create_frame(step,ax,data):
    ax=ax.cla()
    sns.heatmap(data[step][-1].cpu(),annot=True,cmap="cubehelix",ax=ax,cbar=False)
    plt.title('Epoch {} validation on {}'.format(step,exp)  )

#train_loaders_paths=[ "./data/SplitMnist/train/{}".format(x) for x in range(10)]
#test_loaders_paths=[ "./data/SplitMnist/test/{}".format(x) for x in range(10)]




if __name__ == "__main__":
    torch.manual_seed(887)
    L_activations=["relu","gelu","tanh"]
    L_inits=["kaiming_uniform"]

    All=list(range(10))
    patience=3
    Margin=0.05
    for i in range(len(L_activations)):
        for t,exp in enumerate(ListExperiences):
            L_train_acc=[]
            L_train_loss=[]
            L_test_acc_0=[]
            L_test_loss_0=[]
            
            L2=[k for k in All if k not in exp] 
            train_IF=ClassSpecificImageFolder( root="./data/SplitMnist/train/",dropped_classes=L2,transform=transforms.Compose([ transforms.ToTensor(),transforms.Grayscale(1)]))
            T_DL = DataLoader(dataset=train_IF, batch_size=90, num_workers=0, shuffle=True)
            test_IF=ClassSpecificImageFolder( root="./data/SplitMnist/test/",dropped_classes=L2,transform=transforms.Compose([ transforms.ToTensor(),transforms.Grayscale(1)]))
            Ts_DL = DataLoader(dataset=train_IF, batch_size=90, num_workers=0, shuffle=True)

            model = CNN(1,L_activations[i],0,L_inits[0])
            if not(os.path.isdir('./checkpoints/')):
                os.mkdir('./checkpoints/')
            if not(os.path.isdir('./checkpoints/{}/'.format(exp))):
                os.mkdir('./checkpoints/{}/'.format(exp))
            if not(os.path.isdir('./checkpoints/{}/{}'.format(exp,L_activations[i]))):
                os.mkdir('./checkpoints/{}/{}'.format(exp,L_activations[i]))
            if not(os.path.isdir('./checkpoints/{}/{}/metrics/'.format(exp,L_activations[i]))):
                os.mkdir('./checkpoints/{}/{}/metrics/'.format(exp,L_activations[i]))


            # lists to keep track of losses and accuracies
            train_loss, valid_loss_0 = [], [] 
            train_acc, valid_acc_0 = [], []
            train_confus,test_confus =[] , []
            Brain=copy.deepcopy(model)
            Brain=Brain.to(device)

            optimizer = Adam(Brain.parameters(), lr=0.05)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=0.1, step_size_up=1, mode="triangular2", cycle_momentum=False)

            criterion = CrossEntropyLoss()

            valid_epoch_loss0, valid_epoch_acc0,L_mx_st = validate(Brain, Ts_DL,criterion,10)
            valid_loss_0.append(valid_epoch_loss0)
            valid_acc_0.append(valid_epoch_acc0)
            test_confus.append(L_mx_st)
            #print(f"Validation loss: {valid_epoch_loss0:.3f}, validation acc 1: {valid_epoch_acc0:.3f}")

            #print('-'*50)
            # start the training
            stagnate=0
            for epoch in range(40):
                print(f"[INFO]: Epoch {epoch+1} of 40" , t ,L_activations[i])
                train_epoch_loss, train_epoch_acc ,L_mx = train(Brain, T_DL, optimizer, criterion,10)
                train_confus.append(L_mx)
                train_loss.append(train_epoch_loss)
                train_acc.append(train_epoch_acc)
                valid_epoch_loss0, valid_epoch_acc0,L_mx= validate(Brain, Ts_DL,  criterion,10)
                test_confus.append(L_mx)
                valid_loss_0.append(valid_epoch_loss0)
                valid_acc_0.append(valid_epoch_acc0)
                print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
                print(f"Validation loss: {valid_epoch_loss0:.3f}, validation acc 1: {valid_epoch_acc0:.3f}")

                if (valid_epoch_acc0 >60) and (epoch >= 9):
                    if epoch %5==1:
                        torch.save({'epoch': aux,'model_state_dict': Brain.state_dict(),'optimizer_state_dict': optimizer.state_dict(),},'./checkpoints/{}/{}/checkpoint epoch {}.pth'.format(exp,L_activations[i],epoch))
                    if (abs(valid_epoch_acc0-valid_acc_0[-2])<=Margin)  :
                        print("stagnation")
                        stagnate=stagnate+1
                        if stagnate==patience :
                            epoch=-1
                        else:
                            continue
                    else:
                        stagnate=0
                    aux=epoch
                    if (epoch==-1) :
                        aux=epoch
                        break
                print('-'*50)


            #torch.save(Brain.state_dict(),'./checkpoints/{}/{}/{}/checkpoint.pth'.format(ListExperiences[exp_idx][:5],activ,init))
            torch.save({'epoch': aux,'model_state_dict': Brain.state_dict(),'optimizer_state_dict': optimizer.state_dict(),},'./checkpoints/{}/{}/checkpoint.pth'.format(exp,L_activations[i]))
            L_train_acc.append(train_acc)
            L_train_loss.append(train_loss)
            L_test_acc_0.append(valid_acc_0)
            L_test_loss_0.append(valid_loss_0)

            direct='./checkpoints/{}/{}/metrics/'.format(exp,L_activations[i])
            M=np.array(L_test_acc_0) 
            np.save(direct+"Test Accuracy IID.npy", M)

            H=np.array(L_test_loss_0)
            np.save(direct+"Test Loss IID.npy",H)


            N=np.array(L_train_acc)
            np.save(direct+"Train Acc.npy",N)

            O=np.array(L_train_loss)
            np.save(direct+"Train Loss.npy",O)
            
            P=[train_confus[i][-1].cpu() for i in range(len(train_confus))]
            P=torch.stack(P)
            P=np.transpose(P, (1, 2, 0))
            np.save(direct+"multiclass train Confusion matrix raw.npy",P)
            
            W=[test_confus[i][-1].cpu() for i in range(len(test_confus))]
            W=torch.stack(W)
            W=np.transpose(W, (1, 2, 0))
            np.save(direct+"multiclass test Confusion matrix raw.npy",W)
            
            fig, ax = plt.subplots() 
            ax.cla()
            animation = FuncAnimation(fig, create_frame, frames=len(train_confus), fargs=(ax,train_confus))
            wr=PillowWriter(fps=1)
            animation.save('./checkpoints/{}/{}/metrics/Training mcx.gif'.format(exp,L_activations[i]) , writer=wr)
            fig, ax = plt.subplots() 
            ax.cla()
            animation = FuncAnimation(fig, create_frame, frames=len(test_confus), fargs=(ax,test_confus))
            wr=PillowWriter(fps=1)
            animation.save('./checkpoints/{}/{}/metrics/Testing mcx.gif'.format(exp,L_activations[i]) , writer=wr)
