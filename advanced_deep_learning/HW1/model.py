
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F



class SiameseNetwork(nn.Module):

    def __init__(self, in_cannels):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_cannels, 96, kernel_size=11,stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(nn.Flatten(1),
            nn.Linear(31104, 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,30)
        )
        
    def forward_once(self, x):

        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):

        
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
    


if __name__ == '__main__':
    X = torch.rand([1,3,224,224])
    model = SiameseNetwork(3)
    out = model.forward_once(X)
    print(out.shape)