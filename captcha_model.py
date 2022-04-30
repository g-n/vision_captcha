# Computer Vision CAPTCHA Project
# 2020-12-04

import pathlib
from typing import Tuple

import numpy as np
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from PIL import Image
from torch import optim
from torch.tensor import Tensor
import math
import random
from sklearn.metrics import classification_report

class Dataset:
    def __init__(self, root):
        '''
        Class for loading, training, and evaluating models on the FairFace dataset
        :param root: the directory root of the images i.e. ~/FairFace/
        :param size: number of images to use
        '''
        self.root = pathlib.Path(root)
        self.labels = []
        self.to_tensor = torchvision.transforms.ToTensor()
        self.scaler = torchvision.transforms.Resize(100)


        self.root = pathlib.Path(root)
        self.map = {}
        l = []
        labels = []
        self.counts = {
            'Bicycle': 0,
            'Tlight': 0,
            'Cross': 0,
            'Bus': 0,
            'Palm': 0,
            'Bridge': 0,
            'Hydrant': 0,
            'Car': 0,
            'Other': 0,
        }
        self.map = {
            'Bicycle': 0,
            'Bridge': 1,
            'Bus': 2,
            'Car': 3,
            'Cross': 4,
            'Hydrant': 5,
            'Other': 6,
            'Palm': 7,
            'Tlight': 8,
        }
        for qq in self.counts.keys():
            folder = (self.root / 'Large') / qq
            for file in folder.iterdir():
                label = file.name.split(' ')[0]
                self.counts[label] = self.counts[label] + 1
                num = int(file.name[file.name.find('(') + 1:file.name.find(')')])
                labels.append((file, label))
        for label in labels:
            self.labels.append((label[0], label[1], self.map[label[1]]))


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        """
        Returns an image and label for a given index
        :param idx: the numerical index i.e. 0 for train/1.jpg
        :return: a tuple containing tensors of the image and label
        """
        # img = Image.open(self.root / item)
        # return self.to_tensor(img).unsqueeze(0), torch.tensor([self.labels[item]]).view((1, 1))
        filepath, m, target = self.labels[idx]
        # file = self.root / target['filename']
        img = Image.open(filepath).convert('RGB')
        return self.scaler(self.to_tensor(img)).unsqueeze(0), torch.tensor([target])

    def epoch(self, model, dataset, lossfn, optimizer):
        """
        Runs a single epoch on a model
        :param model: pytorch model
        :param dataset: list of keys to iterate the dataset
        :param lossfn: the loss function for training
        :param optimizer: the optimizer to be used
        :return: the losses for a given epoch
        """
        losses = []
        for row in dataset:
            img, target = self.read(row)
            output = model(img)
            loss = lossfn(output, target)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
        return losses

    def split(self, p=0.80):
        random.Random(5).shuffle(self.labels)
        split = int((len(self.labels) * p))
        train = self.labels[:split]
        test = self.labels[split:]
        return train, test

    def train(self, model, dataset, lossfn, optimizer, epochs, verbose=False):
        """
        Trains a model on a given dataset
        :param model: pytorch model
        :param dataset: list of keys to iterate the dataset
        :param lossfn: the loss function for training
        :param optimizer: the optimizer to be used
        :param epochs: the number of epochs
        """
        for epoch in range(epochs):
            losses = self.epoch(model, dataset, lossfn, optimizer)
            if verbose:
                print('  Epoch {}: loss={:.4f}'.format(epoch, np.mean(losses)))

    def read(self, row):
        img = Image.open(row[0]).convert('RGB')
        return self.scaler(self.to_tensor(img)).unsqueeze(0), torch.tensor([row[2]])

    def test(self, model, dataset):
        """
        Evaluates the model on the given dataset
        :param model: pytorch model
        :param dataset: list of keys to iterate the dataset
        """
        # labs = []
        labs = []
        preds = []
        for row in dataset:
            img, target = self.read(row)
            output = model(img)
            pred = output.detach().numpy()[0].argmax()
            labs.append(pred)
            preds.append(target)
        report = classification_report(preds, labs, target_names=list(self.map.keys()),zero_division=0)
        print(report)



class representation(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 24, 3)
        self.conv3 = nn.Conv2d(24, 36, 3)
        self.conv4 = nn.Conv2d(36, 24, 3)
        self.conv5 = nn.Conv2d(24, 16, 3)
        self.fc6 = nn.Linear(16, 12)
        self.fc7 = nn.Linear(12, 9)
        # self.fcP = nn.Linear(8, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        x = F.relu(self.fc6(x.view(-1, 16)))
        x = F.relu(self.fc7(x))
        return x


a = '/data/vision/recaptcha-dataset-master/'
k = representation()
epochs = 30

optimizer = optim.RMSprop(k.parameters(), 0.0001)
ds = Dataset(a)
train, test = ds.split(0.80)
loss_fn = nn.CrossEntropyLoss()
ds.train(k, train, loss_fn, optimizer, epochs, True)
ds.test(k, test)
