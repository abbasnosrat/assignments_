from model import *
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader


def main():
    path = "/home/abbas/Documents/assignments/ADL/HW2/cycleGan/"
    train_set = Horse2Zebra(f"{path}Dataset/trainA",
                            f"{path}Dataset/trainB/", True)
    val_set = Horse2Zebra(f"{path}Dataset/testA/",
                          f"{path}Dataset/testB/", False)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    trainer = Trainer()
    trainer.fit(loader=[train_loader, val_loader], epochs=20)
    plot_logs(trainer.logs)


if __name__ == '__main__':
    main()
