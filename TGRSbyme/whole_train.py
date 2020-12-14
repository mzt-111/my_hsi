from model import HGNN
from data_prepare import data_prepare, data_prepare_whole
from utils import random_mini_batches_GCN
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import metrics
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    setup_seed(20)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_class = 14
    #load data
    img_whole, whole_gt, G_whole, mask_TR, mask_TE = data_prepare_whole(num_class=n_class)
    img_whole = torch.Tensor(img_whole).to(device)
    whole_gt = torch.Tensor(whole_gt).to(device)
    G_whole = torch.Tensor(G_whole).to(device)
    mask_TR = torch.Tensor(mask_TR).to(device)
    mask_TE = torch.Tensor(mask_TE).to(device)

    model = HGNN(in_ch=img_whole.shape[1],
                 n_class=n_class,
                 n_hid=128,
                 dropout=0)
    model.to(device)
    num_epochs = 200
    minibatch_size = 32
    lr = 0.001
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
        (m, n_x) = img_whole.shape
        (m, n_y) = whole_gt.shape
        epoch_loss = 0.
        epoch_acc = 0.
        model.train()
        optimizer.zero_grad()
        output = model(img_whole, G_whole)
        _, label = torch.max(whole_gt, 1)
        label_tr = label[mask_TR>0]
        label_te = label[mask_TE>0]
        output_tr = output[mask_TR>0,:]
        epochloss = criterion(output_tr, label_tr)
        epochloss.backward()
        optimizer.step()
        scheduler.step()
        #calculate accuracy per batch
        _, pre = torch.max(output_tr , 1)
        num_correct = torch.eq(pre, label_tr).sum().float().item()
        #print(num_correct)0.716245
        accuracy = num_correct/label_tr.shape[0]
        epoch_loss = epochloss
        epoch_acc = accuracy


        if epoch % 10 == 0:
            model.eval()
            output_test = model(img_whole, G_whole)
            output_te = output_test[mask_TE>0,:]
            epoch_loss_test = criterion(output_te, label_te)
            _, pre = torch.max(output_te, 1)
            num_correct = torch.eq(pre, label_te).sum().float().item()
            epoch_acc_test = num_correct / label_te.shape[0]
            print("epoch %i: Train_loss: %f, Val_loss: %f, Train_acc: %f, Val_acc: %f" % (epoch, epoch_loss, epoch_loss_test, epoch_acc, epoch_acc_test))

            if epoch % 70 == 0:
                model.eval()
                output_test = model(img_whole, G_whole)
                output_te = output_test[mask_TE > 0, :]
                epoch_loss_test = criterion(output_te, label_te)
                _, pre = torch.max(output_te, 1)
                kappa = metrics.cohen_kappa_score(pre.cpu(), label_te.cpu())
                print(kappa)

        # if epoch % 50 == 0:
        #     costs.append(epoch_loss)
        #     train_acc.append(epoch_acc)
        #     costs_dev.append(epoch_loss_test)
        #     val_acc.append(epoch_acc_test)
        #     # plot the cost
        #     plt.plot(np.squeeze(costs))
        #     plt.plot(np.squeeze(costs_dev))
        #     plt.ylabel('cost')
        #     plt.xlabel('iterations (per tens)')
        #     plt.title("Learning rate =" + str(lr))
        #     plt.show()









