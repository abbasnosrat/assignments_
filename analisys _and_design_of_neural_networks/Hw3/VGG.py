

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchmetrics
from torch.nn import Conv2d, MaxPool2d, Dropout2d,Dropout, ReLU, Flatten, BatchNorm2d, BatchNorm1d, Linear
from torch.utils.data import DataLoader
from tqdm.auto import trange,tqdm
def main():
        device = "cuda:0"





        layers =[Conv2d(3,64, (3, 3), padding=(1, 1)),
                ReLU(),
                BatchNorm2d(64),
                Dropout(0.3),
                Conv2d(64,64, (3, 3), padding=(1, 1)),
                ReLU(),
                BatchNorm2d(64),
                MaxPool2d((2, 2)),
                Conv2d(64,128, (3, 3), padding=(1, 1)),
                ReLU(),
                BatchNorm2d(128),
                Dropout2d(0.4),
                Conv2d(128,128, (3, 3), padding=(1, 1)),
                ReLU(),
                BatchNorm2d(128),
                MaxPool2d((2, 2)),
                Conv2d(128,256, (3, 3), padding=(1, 1)),
                ReLU(),
                BatchNorm2d(256),
                Dropout2d(0.4),
                Conv2d(256,256, (3, 3), padding=(1, 1)),
                ReLU(),
                BatchNorm2d(256),
                Dropout(0.4),
                Conv2d(256,256, (3, 3), padding=(1, 1)),
                ReLU(),
                BatchNorm2d(256),
                MaxPool2d((2, 2)),
                Conv2d(256,512, (3, 3), padding=(1, 1)),
                ReLU(),
                BatchNorm2d(512),
                Dropout2d(0.4),
                Conv2d(512,512, (3, 3), padding=(1, 1)),
                ReLU(),
                BatchNorm2d(512),
                Dropout2d(0.4),
                Conv2d(512,512, (3, 3), padding=(1, 1)),
                ReLU(),
                BatchNorm2d(512),
                MaxPool2d((2, 2)),
                Conv2d(512,512, (3, 3), padding=(1, 1)),
                ReLU(),
                BatchNorm2d(512),
                Dropout2d(0.4),
                Conv2d(512,512, (3, 3), padding=(1, 1)),
                ReLU(),
                BatchNorm2d(512),
                Dropout2d(0.4),
                Conv2d(512,512, (3, 3), padding=(1, 1)),
                ReLU(),
                BatchNorm2d(512),
                MaxPool2d((2, 2)),
                Dropout2d(0.5),
                Flatten(),
                Linear(512,512),
                ReLU(),
                BatchNorm1d(num_features=512),
                Dropout(0.5),
                Linear(512,10)
                ]

        model = nn.Sequential(*layers).to(device)


        ds = datasets.CIFAR10(".",download=True)
        m=((ds.data/255).mean((0,1,2)))
        s = ((ds.data/255).std((0,1,2)))
        print(m,s)




        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(m,s)])
        train_set = datasets.CIFAR10(".",download=True,transform=transform,target_transform=torch.tensor)
        val_set = datasets.CIFAR10(".",download=True,transform=transform,train=False,target_transform=torch.tensor)
        train_loader = DataLoader(train_set,batch_size=128,shuffle=True,num_workers=4)
        val_loader = DataLoader(val_set,batch_size=128,shuffle=False,num_workers=4)




        epochs = 40
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9,verbose=True)
        accuracy = torchmetrics.Accuracy().to(device)
        t_loss = []
        v_loss = []
        t_acc = []
        v_acc = []




        for epoch in trange(epochs):
            model.train()
            r_loss = 0
            r_acc = 0
            for X,y in tqdm(train_loader,leave=False):
                X,y = X.to(device),y.to(device)
                out = model(X)
                loss = loss_fn(out,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                r_loss+= loss.item()
                with torch.no_grad():
                    r_acc +=accuracy(torch.argmax(out,1),y).item()
            t_loss.append(r_loss/len(train_loader))
            t_acc.append(r_acc/len(train_loader))
            model.eval()
            r_loss = 0
            r_acc = 0
            with torch.no_grad():
                for X,y in tqdm(val_loader,leave=False):
                    X,y = X.to(device),y.to(device)
                    out = model(X)
                    loss = loss_fn(out,y)
                    r_loss+= loss.item()
                    r_acc +=accuracy(torch.argmax(out,1),y).item()
            v_loss.append(r_loss/len(val_loader))
            v_acc.append(r_acc/len(val_loader))
            print(f"{epoch+1}: loss: train = {t_loss[-1]}, val = {v_loss[-1]},accuracy: train = {t_acc[-1]}, val = {v_acc[-1]}")
            scheduler.step()




        torch.save({"model":model.state_dict()},"model.pt")




        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(t_loss,label="train")
        plt.plot(v_loss,label="val")
        plt.title("loss")
        plt.subplot(2,1,2)
        plt.plot(t_acc,label="train")
        plt.plot(v_acc,label="val")
        plt.title("accuracy")
        plt.savefig("logs.jpg")
        plt.show()
if __name__ == '__main__' :
    main()