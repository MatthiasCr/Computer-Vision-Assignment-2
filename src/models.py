import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """Single modal Conv Net for 4 Input channels and a binary output"""
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
    

class LateFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # own Net per modality
        self.rgb_net = Net()
        self.xyz_net = Net()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 1)

    def get_embedding_size(self):
        # late fusion uses own embeddings for both modalities
        return ""

    def forward(self, x_img, x_xyz):
        x_rgb = self.rgb_net(x_img)
        x_xyz = self.xyz_net(x_xyz)
        x = torch.cat((x_rgb, x_xyz), 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Embedder(nn.Module):
    def __init__(self, embedder_type="maxPool"):
        super().__init__()
        kernel_size = 3
        self.conv1 = nn.Conv2d(4, 25, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(25, 50, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(50, 100, kernel_size, padding=1)
        
        match embedder_type:
            case "maxPool":
                self.pool = nn.MaxPool2d(2)
            case "strided":
                self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x
    

class IntermediateFusionNet(nn.Module):
    def __init__(self, fusion_type="cat", embedder_type="maxPool"):
        super().__init__()
        self.fusion_type = fusion_type

        self.rgb_embedder = Embedder(embedder_type)
        self.xyz_embedder = Embedder(embedder_type)

        # concatenation doubles the output channels
        if fusion_type == "cat":
            fused_channels = 200
        else:
            fused_channels = 100
        # embedding size is here the size after fusion and before linear layers
        self.embedding_size = fused_channels * 8 * 8

        self.fc1 = nn.Linear(self.embedding_size, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 1)

    def get_embedding_size(self):
        return self.embedding_size

    def forward(self, x_rgb, x_xyz):
        x_rgb = self.rgb_embedder(x_rgb)
        x_xyz = self.xyz_embedder(x_xyz)
        
        match self.fusion_type:
            case "cat":
                x = torch.cat((x_rgb, x_xyz), 1)
            case "add":
                x = torch.add(x_rgb, x_xyz)
            case "had":
                x = torch.mul(x_rgb, x_xyz)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
