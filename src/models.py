import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 3
        self.conv1 = nn.Conv2d(4, 50, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(50, 100, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(100, 200, kernel_size, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(200 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 100)
        
        # single output for binary class probability
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class LateFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # own encoder per modality
        self.rgb_net = Net()
        self.xyz_net = Net()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x_img, x_xyz):
        x_rgb = self.rgb_net(x_img)
        x_xyz = self.xyz_net(x_xyz)
        x = torch.cat((x_rgb, x_xyz), 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Embedder(nn.Module):
    def __init__(self, embedderType="maxPool"):
        super().__init__()
        kernel_size = 3
        self.conv1 = nn.Conv2d(4, 25, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(25, 50, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(50, 100, kernel_size, padding=1)
        
        if embedderType == "maxPool":
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x
    

class IntermediateFusionNet(nn.Module):
    def __init__(self, fusionType="cat", embedderType="maxPool"):
        super().__init__()
        self.fusionType = fusionType

        self.rgb_embedder = Embedder(embedderType)
        self.xyz_embedder = Embedder(embedderType)

        # concatenation doubles the output channels
        if fusionType == "cat":
            fusedOutputs = 200
        else:
            fusedOutputs = 100

        self.fc1 = nn.Linear(fusedOutputs * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x_rgb, x_xyz):
        x_rgb = self.rgb_embedder(x_rgb)
        x_xyz = self.xyz_embedder(x_xyz)
        
        match self.fusionType:
            case "cat":
                x = torch.cat((x_rgb, x_xyz), 1)
            case "add":
                x = torch.add(x_rgb, x_xyz)
            case "hadamard":
                x= torch.mul(x_rgb, x_xyz)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
