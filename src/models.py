import torch
import torch.nn as nn
import torch.nn.functional as F

#
# Models for Task 3-4
#

class LateEmbedder(nn.Module):
    def __init__(self, embedding_size=100, embedder_type="maxpool"):
        super().__init__()
        kernel_size = 3

        if embedder_type == "maxpool":
            stride = 1
            self.pool = nn.MaxPool2d(2)
        elif embedder_type == "strided":
            stride = 2
            # no extra pooling layer
            self.pool = nn.Identity()

        self.conv1 = nn.Conv2d(4, 25, kernel_size, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(25, 50, kernel_size, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(50, 100, kernel_size, padding=1, stride=stride)
        self.fc = nn.Linear(100 * 8 * 8, embedding_size)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


class LateFusionNet(nn.Module):
    def __init__(self, embedder_type="maxpool"):
        super().__init__()
        self.rgb_net = LateEmbedder(100, embedder_type)
        self.xyz_net = LateEmbedder(100, embedder_type)
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x_img, x_xyz):
        x_rgb = self.rgb_net(x_img)
        x_xyz = self.xyz_net(x_xyz)
        x = torch.cat((x_rgb, x_xyz), 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class IntermediateEmbedder(nn.Module):
    def __init__(self, embedder_type="maxpool"):
        super().__init__()
        kernel_size = 3

        if embedder_type == "maxpool":
            stride = 1
            self.pool = nn.MaxPool2d(2)
        elif embedder_type == "strided":
            stride = 2
            # no extra pooling layer
            self.pool = nn.Identity()

        self.conv1 = nn.Conv2d(4, 50, kernel_size, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(50, 100, kernel_size, padding=1, stride=stride)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # no flatten
        # returns shape (B, 100, 16, 16)
        return x 
    

class IntermediateFusionNet(nn.Module):
    def __init__(self, fusion_type="cat", embedder_type="maxpool"):
        super().__init__()
        self.fusion_type = fusion_type

        self.rgb_embedder = IntermediateEmbedder(embedder_type)
        self.xyz_embedder = IntermediateEmbedder(embedder_type)

        # concatenation doubles the output channels
        fused_channels = 200 if fusion_type == "cat" else 100
        
        # shared layers

        if embedder_type == "maxpool":
            stride = 1
            self.pool = nn.MaxPool2d(2)
        elif embedder_type == "strided":
            stride = 2
            # no extra pooling layer
            self.pool = nn.Identity()

        self.conv3 = nn.Conv2d(fused_channels, 100, kernel_size=3, padding=1, stride=stride)
        self.fc1 = nn.Linear(100 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 1)

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
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#
# Models for Task 5
#

class LidarClassifier(nn.Module):
    def __init__(self, emb_size: int = 200, normalize_embs: bool = True):
        super().__init__()
        kernel_size = 3
        n_classes = 1
        self.embedding_size = emb_size
        self.normalize_embs = normalize_embs

        self.embedder = nn.Sequential(
            nn.Conv2d(4, 50, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(100, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(200, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, self.embedding_size)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_size, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes)
        )

    def get_embedding_size(self):
        return self.embedding_size

    def get_embs(self, lidar_xyz):
        embs = self.embedder(lidar_xyz)
        if self.normalize_embs:
            embs = F.normalize(embs, dim=1)
        return embs
    
    def forward(self, raw_data=None, data_embs=None):
        assert (raw_data is not None or data_embs is not None), "No Lidar or embeddings given."
        if raw_data is not None:
            data_embs = self.get_embs(raw_data)
        return self.classifier(data_embs)


class CILPEmbedder(nn.Module):
    def __init__(self, in_ch, emb_size=200):
        super().__init__()
        kernel_size = 3
        stride = 1
        padding = 1

        # Convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 50, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(100, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(200, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # Embeddings
        self.dense_emb = nn.Sequential(
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, emb_size)
        )

    def forward(self, x):
        conv = self.conv(x)
        emb = self.dense_emb(conv)
        return F.normalize(emb, dim=1)


class ContrastivePretraining(nn.Module):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.embedding_size = 200

        self.img_embedder = CILPEmbedder(4, self.embedding_size)
        self.lidar_embedder = CILPEmbedder(4, self.embedding_size)
        self.cos = nn.CosineSimilarity()

        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07))

    def get_embedding_size(self):
        return self.embedding_size

    def forward(self, rgb_imgs, lidar_xyz):
        img_emb = self.img_embedder(rgb_imgs)
        lidar_emb = self.lidar_embedder(lidar_xyz)
        img_emb = F.normalize(img_emb, dim=1)
        lidar_emb = F.normalize(lidar_emb, dim=1)

        repeated_img_emb = img_emb.repeat_interleave(len(img_emb), dim=0)
        repeated_lidar_emb = lidar_emb.repeat(len(lidar_emb), 1)

        similarity = self.cos(repeated_img_emb, repeated_lidar_emb)
        similarity = torch.unflatten(similarity, 0, (self.batch_size, self.batch_size))
        
        logits_per_img = self.logit_scale.exp() * similarity

        logits_per_img = similarity
        logits_per_lidar = similarity.T
        return logits_per_img, logits_per_lidar


class Projector(nn.Module):
    def __init__(self, img_emb_size, lidar_emb_size, normalize_output: bool = True):
        super().__init__()
        self.normalize_output = normalize_output
        self.layers = nn.Sequential(
            nn.Linear(img_emb_size, 100),
            nn.ReLU(),
            nn.Linear(100, lidar_emb_size)
        )
    def forward(self, img_emb):
        proj = self.layers(img_emb)
        if self.normalize_output:
            proj = F.normalize(proj, dim=1)
        return proj


class RGB2LiDARClassifier(nn.Module):
    def __init__(self, projector, cilp_model, classifier):
        super().__init__()
        self.projector = projector
        self.img_embedder = cilp_model.img_embedder
        self.shape_classifier = classifier
    
    def forward(self, imgs):
        img_encodings = self.img_embedder(imgs)
        proj_lidar_embs = self.projector(img_encodings)
        return self.shape_classifier(data_embs=proj_lidar_embs)
