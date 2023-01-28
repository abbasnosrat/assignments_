import torch
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from model import *
from tqdm.auto import tqdm


class Horse2Zebra(Dataset):
    def __init__(self, root_A, root_B, train=True):
        self.root_A = root_A
        self.root_B = root_B
        if train:
            self.transform = A.Compose([
                A.Resize(width=256, height=256),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                            0.5, 0.5, 0.5], max_pixel_value=255),
                ToTensorV2(),
            ], additional_targets={"image0": "image"})
        else:
            self.transform = A.Compose([
                A.Resize(width=256, height=256),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                            0.5, 0.5, 0.5], max_pixel_value=255),
                ToTensorV2(),
            ], additional_targets={"image0": "image"})

        self.A_images = os.listdir(root_A)
        self.B_images = os.listdir(root_B)
        self.length_dataset = max(len(self.A_images), len(self.B_images))
        self.A_len = len(self.A_images)
        self.B_len = len(self.B_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        A_img = self.A_images[index %
                              self.A_len if self.length_dataset == self.B_len else index]
        B_img = self.B_images[index %
                              self.B_len if self.length_dataset == self.A_len else index]

        A_path = os.path.join(self.root_A, A_img)
        B_path = os.path.join(self.root_B, B_img)

        A_img = np.array(Image.open(A_path).convert("RGB"))
        B_img = np.array(Image.open(B_path).convert("RGB"))

        transformed = self.transform(image=A_img, image0=B_img)
        A_img = transformed["image"]
        B_img = transformed["image0"]

        return A_img, B_img


class Trainer:
    def __init__(self, lr=1e-5, beta=(0.5, 0.999), cycle_coef=0.8, path="/home/abbas/Documents/assignments/ADL/HW2/cycleGan/"):
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.cycle_coef = cycle_coef
        self.path = path
        self.genA = Generator()
        self.genB = Generator()
        self.discA = Discriminator()
        self.discB = Discriminator()
        self.genA = self.genA.to(self.device)
        self.genB = self.genB.to(self.device)
        self.discA = self.discA.to(self.device)
        self.discB = self.discB.to(self.device)
        self.optGen = torch.optim.AdamW(
            list(self.genA.parameters())+list(self.genB.parameters()), lr=lr, betas=beta)
        self.optDisc = torch.optim.AdamW(
            list(self.discA.parameters())+list(self.discB.parameters()), lr=lr, betas=beta)
        self.l1 = nn.L1Loss()  # identity loss
        self.mse = nn.MSELoss()

        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()

    def training_step(self, loader):
        ADV_losses = []
        cycle_losses = []
        D_losses = []
        G_losses = []
        self.genA.train()
        self.genB.train()
        self.discA.train()
        self.discB.train()
        loop = tqdm(loader, leave=True)
        idx = 0
        for A, B in loop:
            idx += 1
            A = A.to(self.device)
            B = B.to(self.device)
            # train discs
            with torch.cuda.amp.autocast():  # 16 bit
                fake_A = self.genA(B)
                D_A_real = self.discA(A)
                D_A_fake = self.discA(fake_A.detach())
                D_A_real_loss = self.mse(D_A_real, torch.ones_like(D_A_real))
                D_A_fake_loss = self.mse(D_A_fake, torch.zeros_like(D_A_fake))
                D_A_loss = D_A_real_loss+D_A_fake_loss

                fake_B = self.genB(A)
                D_B_real = self.discB(B)
                D_B_fake = self.discB(fake_B.detach())
                D_B_real_loss = self.mse(D_B_real, torch.ones_like(D_B_real))
                D_B_fake_loss = self.mse(D_B_fake, torch.zeros_like(D_B_fake))
                D_B_loss = D_B_real_loss+D_B_fake_loss

                D_loss = (D_A_loss+D_B_loss)/2

            self.optDisc.zero_grad()
            self.d_scaler.scale(D_loss).backward()
            self.d_scaler.step(self.optDisc)
            self.d_scaler.update()

            with torch.cuda.amp.autocast():
                D_A_fake = self.discA(fake_A)
                D_B_fake = self.discB(fake_B)
                G_A_ADV_loss = self.mse(D_A_fake, torch.ones_like(D_A_fake))
                G_B_ADV_loss = self.mse(D_B_fake, torch.ones_like(D_B_fake))
                ADV_loss = G_A_ADV_loss+G_B_ADV_loss

                cycle_A = self.genA(fake_B)
                cycle_B = self.genB(fake_A)
                cycle_A_loss = self.l1(A, cycle_A)
                cycle_B_loss = self.l1(B, cycle_B)
                cycle_loss = cycle_A_loss+cycle_B_loss

                G_loss = ADV_loss+(cycle_loss*self.cycle_coef)
            self.optGen.zero_grad()
            self.g_scaler.scale(G_loss).backward()
            self.g_scaler.step(self.optGen)
            self.g_scaler.update()
            with torch.no_grad():
                D_loss = D_loss.item()
                ADV_loss = ADV_loss.item()
                cycle_loss = cycle_loss.item()
                G_loss = G_loss.item()
                ADV_losses.append(ADV_loss)
                cycle_losses.append(cycle_loss)
                D_losses.append(D_loss)
                G_losses.append(G_loss)

            loop.set_postfix(dict(D_loss=D_loss, ADV_loss=ADV_loss,
                             cycle_loss=cycle_loss, G_loss=G_loss))
            loop.set_description("training step")
            if idx % 200 == 0:
                torchvision.utils.save_image(
                    fake_A*0.5+0.5, f"{self.path}/figures/train/A_epoch{self.epoch}_{idx}.png")
                torchvision.utils.save_image(
                    fake_B*0.5+0.5, f"{self.path}/figures/train/B_epoch{self.epoch}_{idx}.png")
        return np.mean(D_losses), np.mean(ADV_losses), np.mean(cycle_losses), np.mean(G_losses)

    def validation_step(self, loader):
        ADV_losses = []
        cycle_losses = []
        D_losses = []
        G_losses = []
        self.genA.eval()
        self.genB.eval()
        self.discA.eval()
        self.discB.eval()
        loop = tqdm(loader, leave=True)
        idx = 0
        self.model.eval()
        with torch.no_grad():
            for A, B in loop:
                idx += 1
                A = A.to(self.device)
                B = B.to(self.device)
                # train discs
                with torch.cuda.amp.autocast():  # 16 bit
                    fake_A = self.genA(B)
                    D_A_real = self.discA(A)
                    D_A_fake = self.discA(fake_A.detach())
                    D_A_real_loss = self.mse(
                        D_A_real, torch.ones_like(D_A_real))
                    D_A_fake_loss = self.mse(
                        D_A_fake, torch.zeros_like(D_A_fake))
                    D_A_loss = D_A_real_loss+D_A_fake_loss

                    fake_B = self.genB(A)
                    D_B_real = self.discB(B)
                    D_B_fake = self.discB(fake_B.detach())
                    D_B_real_loss = self.mse(
                        D_B_real, torch.ones_like(D_B_real))
                    D_B_fake_loss = self.mse(
                        D_B_fake, torch.zeros_like(D_B_fake))
                    D_B_loss = D_B_real_loss+D_B_fake_loss

                    D_loss = (D_A_loss+D_B_loss)/2

                with torch.cuda.amp.autocast():
                    D_A_fake = self.discA(fake_A)
                    D_B_fake = self.discB(fake_B)
                    G_A_ADV_loss = self.mse(
                        D_A_fake, torch.ones_like(D_A_fake))
                    G_B_ADV_loss = self.mse(
                        D_B_fake, torch.ones_like(D_B_fake))
                    ADV_loss = G_A_ADV_loss+G_B_ADV_loss

                    cycle_A = self.genA(fake_B)
                    cycle_B = self.genB(fake_A)
                    cycle_A_loss = self.l1(A, cycle_A)
                    cycle_B_loss = self.l1(B, cycle_B)
                    cycle_loss = cycle_A_loss+cycle_B_loss

                    G_loss = ADV_loss+(cycle_loss*self.cycle_coef)

                D_loss = D_loss.item()
                ADV_loss = ADV_loss.item()
                cycle_loss = cycle_loss.item()
                G_loss = G_loss.item()
                ADV_losses.append(ADV_loss)
                cycle_losses.append(cycle_loss)
                D_losses.append(D_loss)
                G_losses.append(G_loss)

                loop.set_postfix(dict(D_loss=D_loss, ADV_loss=ADV_loss,
                                      cycle_loss=cycle_loss, G_loss=G_loss))
                loop.set_description("validation step")
                if idx % 28 == 0:
                    torchvision.utils.save_image(
                        fake_A*0.5+0.5, f"{self.path}/figures/val/A_epoch{self.epoch}_{idx}.png")
                    torchvision.utils.save_image(
                        fake_B*0.5+0.5, f"{self.path}/figures/val/B_epoch{self.epoch}_{idx}.png")
        return np.mean(D_losses), np.mean(ADV_losses), np.mean(cycle_losses), np.mean(G_losses)

    def fit(self, loader, epochs):

        log_dict = dict(D_loss=[], ADV_loss=[], cycle_loss=[],
                        G_loss=[], val_D_loss=[], val_ADV_loss=[], val_cycle_loss=[],
                        val_G_loss=[])
        for self.epoch in range(epochs):
            D_loss, ADV_loss, cycle_loss, G_loss = self.training_step(
                loader[0])
            log_dict['D_loss'].append(D_loss)
            log_dict['ADV_loss'].append(ADV_loss)
            log_dict['cycle_loss'].append(cycle_loss)
            log_dict['G_loss'].append(G_loss)

            D_loss, ADV_loss, cycle_loss, G_loss = self.training_step(
                loader[1])
            log_dict['val_D_loss'].append(D_loss)
            log_dict['val_ADV_loss'].append(ADV_loss)
            log_dict['val_cycle_loss'].append(cycle_loss)
            log_dict['val_G_loss'].append(G_loss)
            self.logs = pd.DataFrame(log_dict)
            print(self.logs.tail(1))
        self.logs.to_csv("logs.csv")


def plot_logs(logs, path="/home/abbas/Documents/assignments/ADL/HW2/cycleGan/"):
    plt.figure(figsize=[20, 20])

    plt.subplot(2, 2, 1)
    plt.plot(logs['D_loss'], label="train")
    plt.plot(logs['val_D_loss'], label="validation")
    plt.legend()
    plt.title("discriminator loss")

    plt.subplot(2, 2, 2)
    plt.plot(logs['ADV_loss'], label="train")
    plt.plot(logs['val_ADV_loss'], label="validation")
    plt.legend()
    plt.title("adversarial loss")

    plt.subplot(2, 2, 3)
    plt.plot(logs['cycle_loss'], label="train")
    plt.plot(logs['val_cycle_loss'], label="validation")
    plt.legend()
    plt.title("cycle loss")

    plt.subplot(2, 2, 4)
    plt.plot(logs['G_loss'], label="train")
    plt.plot(logs['val_G_loss'], label="validation")
    plt.legend()
    plt.title("generator loss")
    plt.savefig(f"{path}/figures/logs/training_logs.jpg")
