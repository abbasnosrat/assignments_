import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchmetrics import Accuracy
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import trange, tqdm
import os


def SI_evaluation(outs, ys):
    with torch.no_grad():
        assert ys.get_device() != -1
        assert outs.get_device() != -1
        ys = ys
        m = outs @ outs.T
        d = torch.diag(m)
        d = d.reshape(-1, 1)
        w = torch.tile(d, (1, outs.shape[0]))
        D = w+w.T - 2*m
        inf = torch.max(D)*100
        I = inf*torch.eye(D.shape[0]).cuda()
        D = D+I
        labs = torch.argmin(D, dim=1)
        labs = labs.detach().cpu().numpy()
        labs = ys[labs]
        return (labs == ys).sum()/len(labs)


def forward_selection(X, y):

    selected = []
    best_SIs = []
    done = False
    prv_SI = 0
    while not done:
        SI_list = []
        idx_list = []
        for i in trange(X.shape[1], leave=False):
            idx = selected.copy()
            if not i in idx:
                idx.append(i)
                features = X[:, idx]
                SI = SI_evaluation(features, y)
                SI_list.append(SI.item())
                idx_list.append(i)

        best_idx = np.argmax(SI_list)
        best_SI = SI_list[best_idx]
        best_feature = idx_list[best_idx]
        print(f"best SI:{best_SI} for {best_feature}")
        if best_SI <= prv_SI:
            print("best features obtaied")
            done = True
        else:
            selected.append(best_feature)
            best_SIs.append(best_SI)
            prv_SI = best_SI
    return selected, best_SIs
