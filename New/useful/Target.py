import torch
import torchvision
from itertools import product
from itertools import combinations
#from avalanche.benchmarks.datasets import MNIST, FashionMNISTKMNIST, EMNIST, \
#QMNIST, FakeData, CocoCaptions, CocoDetection, LSUN, ImageNet, CIFAR10, \
#CIFAR100, STL10, SVHN, PhotoTour, SBU, Flickr8k, Flickr30k, VOCDetection, \
#VOCSegmentation, Cityscapes, SBDataset, USPS, HMDB51, UCF101, \
#CelebA, CORe50Dataset, TinyImagenet, CUB200, OpenLORIS
from avalanche.benchmarks.classic import SplitMNIST
#from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.generators import dataset_benchmark
# \tensors_benchmark, paths_benchmark,filelist_benchmark, 


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from torchvision import transforms
from torchvision.datasets import ImageFolder

#from avalanche.training.supervised import Naive,  Replay, Cumulative,EWC,GenerativeReplay ,SynapticIntelligence# ,CWRStar,GDumb,LwF, GEM, AGEM,
#and many more! https://avalanche-api.continualai.org/en/v0.3.1/training.html#training

#from avalanche.training.plugins import EvaluationPlugin,EarlyStoppingPlugin ,ReplayPlugin
from avalanche.training.determinism.rng_manager import RNGManager


from avalanche.evaluation.metrics import Accuracy, TaskAwareAccuracy
from avalanche.evaluation.metrics import accuracy_metrics, \
    loss_metrics, forgetting_metrics, bwt_metrics,\
    confusion_matrix_metrics, cpu_usage_metrics, \
    disk_usage_metrics, gpu_usage_metrics, MAC_metrics, \
    ram_usage_metrics, timing_metrics
from avalanche.logging import InteractiveLogger,TensorboardLogger
from avalanche.benchmarks.generators import filelist_benchmark, dataset_benchmark, \
                                            tensors_benchmark, paths_benchmark

from sklearn.model_selection import StratifiedKFold
import copy
import dill


from itertools import product
from itertools import combinations


import os
import json
import pandas as pd
from tqdm import tqdm



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
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
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
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


def validate(model, testloader, criterion):
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
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    S=combinations(range(10), 5)
    All=list(range(10))
    ListExperiences=list()
    for i in S :
        L1=list(i)
        L2=[k for k in All if k not in L1] 
        L1.extend(L2)
        ListExperiences.append(L1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds=[22,13,1234,74,5]
    for seed in seeds:
        RNGManager.set_random_seeds(seed)
        torch.manual_seed(seed)
        L_activations=["silu","sigmoid","leakyrelu","relu"] #,"gelu","tanh"
        L_inits=["xavier_uniform","uniform","kaiming_normal","xavier_normal" ,"xavier_uniform","uniform","normal","kaiming_uniform"]
        for activ in L_activations:
            for init in L_inits :
                model = CNN(1,activ,0,init)
                for exp_idx in range(252):
                    patience=5
                    Margin=0.03
                    L_train_acc=[]
                    L_test_acc_0=[]
                    L_test_loss_0=[]
                    L_test_acc_1=[]
                    L_test_loss_1=[]
                    L_train_loss=[]
                    #for i,(train_index,test_index) in tqdm(enumerate(skf.split(split_mnist.train_stream[0].dataset,split_mnist.train_stream[0].dataset.targets))):
                    test_0_dirpath=r"./data/SplitMnist/{}/test 0/".format(ListExperiences[exp_idx][:5])
                    test_1_dirpath=r"./data/SplitMnist/{}/test 1/".format(ListExperiences[exp_idx][:5])
                    dirpath=r"./data/Full MNIST/train/fold 20/"

                    train_IF=ImageFolder(dirpath,transforms.Compose([ transforms.ToTensor(),transforms.Grayscale(1)]))
                    test0_IF=ImageFolder(test_0_dirpath,transform = transforms.Compose([ transforms.ToTensor(),transforms.Grayscale(1) ]) )
                    test1_IF=ImageFolder(test_1_dirpath,transform = transforms.Compose([ transforms.ToTensor(),transforms.Grayscale(1) ]) )

                    test_dataloader_custom0 = DataLoader(dataset=test0_IF, batch_size=40, num_workers=0, shuffle=False)# don't usually need to shuffle testing data# use custom created test Dataset
                    test_dataloader_custom1 = DataLoader(dataset=test1_IF, batch_size=40, num_workers=0, shuffle=False)

                    train_dataloader_custom = DataLoader(dataset=train_IF, batch_size=180, num_workers=0, shuffle=True) 



                    
                    if not(os.path.isdir('./Target/')):
                        os.mkdir('./Target/')
                    if not(os.path.isdir('./Target/{}/'.format(seed))):
                        os.mkdir('./Target/{}/'.format(seed))
                    if not(os.path.isdir('./Target/{}/{}'.format(seed,activ))):
                        os.mkdir('./Target/{}/{}'.format(seed,activ))
                    if not(os.path.isdir('./Target/{}/{}/{}'.format(seed,activ,init))):
                        os.mkdir('./Target/{}/{}/{}'.format(seed,activ,init))
                    if not(os.path.isdir('./Target/{}/{}/{}/'.format(seed,activ,init))):
                        os.mkdir('./Target/{}/{}/{}/'.format(seed,activ,init))

                    # lists to keep track of losses and accuracies
                    train_loss, valid_loss_0 ,valid_loss_1= [], [] ,[]
                    train_acc, valid_acc_0 ,valid_acc_1= [], [], []

                    print('-'*50)
                    # start the training
                    stagnate=0
                    if exp_idx==0:
                        Brain=copy.deepcopy(model)
                        Brain=Brain.to(device)

                        optimizer = Adam(Brain.parameters(), lr=0.5)
                        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.025, max_lr=0.1, step_size_up=1750, mode="triangular2", cycle_momentum=False)

                        criterion_tr = CrossEntropyLoss()
                        criterion_t0 = CrossEntropyLoss()
                        criterion_t1 = CrossEntropyLoss()

                        valid_epoch_loss0, valid_epoch_acc0 = validate(Brain, test_dataloader_custom0,criterion_t0)
                        valid_loss_0.append(valid_epoch_loss0)
                        valid_acc_0.append(valid_epoch_acc0)
                        print(f"Validation loss: {valid_epoch_loss0:.3f}, validation acc 1: {valid_epoch_acc0:.3f}")


                        valid_epoch_loss1, valid_epoch_acc1 = validate(Brain, test_dataloader_custom1,criterion_t1)
                        valid_loss_1.append(valid_epoch_loss1)
                        valid_acc_1.append(valid_epoch_acc1)
                        print(f"Validation loss: {valid_epoch_loss1:.3f}, validation acc 2: {valid_epoch_acc1:.3f}")
                        for epoch in range(70):
                            print(f"[INFO]: Epoch {epoch+1} of 40")
                            train_epoch_loss, train_epoch_acc = train(Brain, train_dataloader_custom, optimizer, criterion_tr)
                            valid_epoch_loss0, valid_epoch_acc0 = validate(Brain, test_dataloader_custom0,  criterion_t0)
                            train_loss.append(train_epoch_loss)
                            valid_loss_0.append(valid_epoch_loss0)
                            train_acc.append(train_epoch_acc)
                            valid_acc_0.append(valid_epoch_acc0)
                            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
                            print(f"Validation loss: {valid_epoch_loss0:.3f}, validation acc 1: {valid_epoch_acc0:.3f}")
                            valid_epoch_loss1, valid_epoch_acc1 = validate(Brain, test_dataloader_custom1,  criterion_t1)

                            valid_loss_1.append(valid_epoch_loss1)
                            valid_acc_1.append(valid_epoch_acc1)
                            print(f"Validation loss: {valid_epoch_loss1:.3f}, validation acc 2: {valid_epoch_acc1:.3f}")
                            if  (epoch > 10) and (valid_epoch_acc0 > 60):
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
                    else :
                        #train_epoch_loss, train_epoch_acc = train(Brain, train_dataloader_custom, optimizer, criterion)
                        valid_epoch_loss0, valid_epoch_acc0 = validate(Brain, test_dataloader_custom0,  criterion)
                        #train_loss.append(train_epoch_loss)
                        valid_loss_0.append(valid_epoch_loss0)
                        #train_acc.append(train_epoch_acc)
                        valid_acc_0.append(valid_epoch_acc0)
                        #print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
                        print(f"Validation loss: {valid_epoch_loss0:.3f}, validation acc 1: {valid_epoch_acc0:.3f}")
                        valid_epoch_loss1, valid_epoch_acc1 = validate(Brain, test_dataloader_custom1,  criterion)
                        print(f"Validation loss: {valid_epoch_loss1:.3f}, validation acc 2: {valid_epoch_acc1:.3f}")
                        valid_loss_1.append(valid_epoch_loss1)
                        valid_acc_1.append(valid_epoch_acc1)
                    
                        

                    #torch.save(Brain.state_dict(),'./checkpoints/{}/{}/{}/checkpoint.pth'.format(ListExperiences[exp_idx][:5],activ,init))
                    torch.save({'epoch': aux,'model_state_dict': Brain.state_dict(),'optimizer_state_dict': optimizer.state_dict(),},'./Target/{}/{}/{}/checkpoint.pth'.format(seed,activ,init))
                    L_train_acc.append(train_acc)
                    L_train_loss.append(train_loss)
                    L_test_acc_0.append(valid_acc_0)
                    L_test_loss_0.append(valid_loss_0)
                    L_test_acc_1.append(valid_acc_1)
                    L_test_loss_1.append(valid_loss_1)

                    if not(os.path.isdir('./checkpoints/{}/{}/'.format(ListExperiences[exp_idx][:5],activ))):
                        os.mkdir('./checkpoints/{}/{}/'.format(ListExperiences[exp_idx][:5],activ))
                    if not('./checkpoints/{}/{}/{}/'.format(ListExperiences[exp_idx][:5],activ,init)):
                        os.mkdir('./checkpoints/{}/{}/{}/'.format(ListExperiences[exp_idx][:5],activ,init))
                    if not('./checkpoints/{}/{}/{}/Target Model/'.format(ListExperiences[exp_idx][:5],activ,init,seed)):
                        os.mkdir('./checkpoints/{}/{}/{}/Target Model/'.format(ListExperiences[exp_idx][:5],activ,init,seed))
                    if not('./checkpoints/{}/{}/{}/Target Model/{}/'.format(ListExperiences[exp_idx][:5],activ,init,seed)):
                        os.mkdir('./checkpoints/{}/{}/{}/Target Model/{}/'.format(ListExperiences[exp_idx][:5],activ,init,seed))

                    direct='./checkpoints/{}/{}/{}/Target Model/{}'.format(ListExperiences[exp_idx][:5],activ,init,seed)


                    M=np.array(L_test_acc_0) 
                    np.save(direct+"Test Accuracy IID.npy", M)
                    
                    H=np.array(L_test_loss_0)
                    np.save(direct+"Test Loss IID.npy",H)
                    
                    W=np.array(L_test_acc_1)
                    np.save(direct+"Test Accuracy OOD.npy",W)
                    
                    V=np.array(L_test_loss_1)
                    np.save(direct+"Test Loss OOD.npy",V)

                    N=np.array(L_train_acc)
                    np.save(direct+"Train Acc.npy",N)
                    
                    O=np.array(L_train_loss)
                    np.save(direct+"Train Loss.npy",O)

