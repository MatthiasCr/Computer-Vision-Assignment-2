import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import fiftyone as fo


def get_torch_xyza(lidar_depth, azimuth, zenith):
    x = lidar_depth * torch.sin(-azimuth[:, None]) * torch.cos(-zenith[None, :])
    y = lidar_depth * torch.cos(-azimuth[:, None]) * torch.cos(-zenith[None, :])
    z = lidar_depth * torch.sin(-zenith[None, :])
    a = torch.where(lidar_depth < 50.0, torch.ones_like(lidar_depth), torch.zeros_like(lidar_depth))
    xyza = torch.stack((x, y, z, a))
    return xyza


class MultimodalDataset(Dataset):
    def __init__(self, fo_dataset, split_tag, img_transform):
        """
        Args:
            fo_dataset: grouped FiftyOne dataset with labels "cube" and "sphere"
            split_tag: "train" or "val" (sample tag on RGB samples)
            transform: torchvision transform for RGB
        """
        self.rgb_data = []
        self.lidar_data = []
        self.labels = []

        # mapping label strings to indices
        self.label_map = {
            "cube": 0,
            "sphere": 1,
        }

        self.azimuth_cubes = torch.tensor(fo_dataset.info["azimuth_cubes"], dtype=torch.float32).view(-1)
        self.zenith_cubes  = torch.tensor(fo_dataset.info["zenith_cubes"], dtype=torch.float32).view(-1)
        self.azimuth_spheres = torch.tensor(fo_dataset.info["azimuth_spheres"], dtype=torch.float32).view(-1)
        self.zenith_spheres = torch.tensor(fo_dataset.info["zenith_spheres"], dtype=torch.float32).view(-1)

        rgb_view = (fo_dataset.select_group_slices("rgb").match_tags(split_tag))
        
        for rgb_sample in rgb_view:
          # Get paired lidar sample via group
          lidar_sample = fo_dataset.get_group(rgb_sample.group.id, "lidar")['lidar']
          
          rgb = Image.open(rgb_sample.filepath)
          rgb = img_transform(rgb)

          lidar_depth = np.load(lidar_sample.filepath)
          lidar_depth = torch.from_numpy(lidar_depth).to(torch.float32)

          label = rgb_sample.label.label

          self.rgb_data.append(rgb)
          self.lidar_data.append(lidar_depth)
          self.labels.append(torch.tensor(self.label_map[label], dtype=torch.float32)[None])

    def __len__(self):
        return len(self.rgb_data)

    def __getitem__(self, idx):
        rgb = self.rgb_data[idx]
        lidar_depth = self.lidar_data[idx]
        label = self.labels[idx]

        if label == 0:
            az = self.azimuth_cubes
            ze = self.zenith_cubes
        else:
            az = self.azimuth_spheres
            ze = self.zenith_spheres

        lidar_xyza = get_torch_xyza(lidar_depth, az, ze)

        return rgb, lidar_xyza, label