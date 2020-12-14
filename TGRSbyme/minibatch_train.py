from model import HGNN
from data_prepare import data_prepare, data_prepare_whole
from utils import random_mini_batches_GCN
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_class = 13
#load data
img_train, img_test, gt_train, gt_test, G_tr, G_te = data_prepare_whole(num_class=n_class)
img_test = torch.Tensor(img_test).to(device)
img_train = torch.Tensor(img_train).to(device)
gt_test = torch.Tensor(gt_test).to(device)
gt_train = torch.Tensor(gt_train).to(device)
G_te = torch.Tensor(G_te).to(device)
G_tr = torch.Tensor(G_tr).to(device)


model = HGNN(in_ch=img_train.shape[1],
             n_class=n_class,
             n_hid=128,
             dropout=0)
model.to(device)
num_epochs = 200
minibatch_size = 32
lr = 0.005
weight_decay = 0.0005
milestones = [50, 100]
gamma = 0.5

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
criterion = torch.nn.CrossEntropyLoss()

costs = []
costs_dev = []
train_acc = []
val_acc = []
seed = 1
for epoch in range(num_epochs):
    (m, n_x) = img_train.shape
    (m, n_y) = gt_train.shape
    epoch_loss = 0.
    epoch_acc = 0.
    num_minibatches = int(m / minibatch_size)
    minibatches = random_mini_batches_GCN(img_train, gt_train, G_tr, minibatch_size, seed=seed)
    seed = seed + 1
    for minibatch in minibatches:
        (batch_x, batch_y, batch_g) = minibatch
        # batch_x = torch.Tensor(batch_x).to(device)
        # batch_y = torch.Tensor(batch_y).to(device)
        # batch_g = torch.Tensor(batch_g).to(device)
        model.train()
        optimizer.zero_grad()
        output = model(batch_x, batch_g)
        _, batch_label = torch.max(batch_y, 1)
        minibatch_loss = criterion(output, batch_label)
        minibatch_loss.backward()
        optimizer.step()
        scheduler.step()
        #calculate accuracy per batch
        _, pre = torch.max(output, 1)
        num_correct = torch.eq(pre, batch_label).sum().float().item()
        #print(num_correct)0.716245
        accuracy = num_correct/batch_label.shape[0]
        epoch_loss += minibatch_loss
        epoch_acc += accuracy

    epoch_loss = epoch_loss/(num_minibatches+ 1)
    epoch_acc = epoch_acc/(num_minibatches+ 1)

    if epoch % 10 == 0:
        model.eval()
        output_test = model(img_test, G_te)
        _, gt_label = torch.max(gt_test, 1)
        epoch_loss_test = criterion(output_test, gt_label)
        _, pre = torch.max(output_test, 1)
        num_correct = torch.eq(pre, gt_label).sum().float().item()
        epoch_acc_test = num_correct / gt_label.shape[0]
        print("epoch %i: Train_loss: %f, Val_loss: %f, Train_acc: %f, Val_acc: %f" % (epoch, epoch_loss, epoch_loss_test, epoch_acc, epoch_acc_test))

    if epoch % 50 == 0:
        costs.append(epoch_loss)
        train_acc.append(epoch_acc)
        costs_dev.append(epoch_loss_test)
        val_acc.append(epoch_acc_test)
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.plot(np.squeeze(costs_dev))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(lr))
        plt.show()









