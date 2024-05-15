import torch
import os
import utils
import json
import random
import glob
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2


class DesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, inl_path, log_input, train=True, avg_path=None, device=None, render_size=256,
                 num_renderings=1):
        self.dataset_path = dataset_path
        self.avg_path = avg_path
        self.log_input = log_input
        self.train = train
        self.device = device
        self.render_size = render_size
        self.num_renderings = num_renderings
        self.inl_path = inl_path

        self.ready()

    def ready(self):
        self.load_path()

    def load_path(self):
        self.paths = glob.glob(self.inl_path + "\\*.png")
        self.dataset_len = len(self.paths)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        path = self.paths[index]
        name = path.split('\\')[-1]
        _, normal, diffuse, roughness, specular = \
            utils.read_image_material_tiled(os.path.join(self.dataset_path, name))

        renderings, _, _, inl, _ = utils.read_img_materials_auxrend(path, split=5)

        start = random.randint(1, 31)
        renderings = torch.FloatTensor(renderings[start: start + 256, start: start + 256, :])
        normal = torch.FloatTensor(normal[start: start + 256, start: start + 256, :])
        diffuse = torch.FloatTensor(diffuse[start: start + 256, start: start + 256, :])
        roughness = torch.FloatTensor(roughness[start: start + 256, start: start + 256, :])
        specular = torch.FloatTensor(specular[start: start + 256, start: start + 256, :])

        materials = torch.FloatTensor(np.stack([normal, diffuse, roughness, specular], axis=0))
        materials = utils.adapt_roughness(materials, torch.device('cpu'))
        materials = utils.material_reshape(materials)  # [256, 256, 12]
        materials = utils.preprocess(materials)  # [0, 1] -> [-1, 1]

        renderings = utils.preprocess(renderings)
        return renderings, materials



class Des18DatasetLoad(torch.utils.data.Dataset):
    def __init__(self, gt_material_path, input_path, log_input):
        self.gt_material_path = gt_material_path
        self.input_path = input_path
        self.log_input = log_input

        self.load_path()

    def load_path(self):
        self.paths = glob.glob(self.input_path + "/*.png")
        self.dataset_len = len(self.paths)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        path = self.paths[index]
        name = path.split('/')[-1]

        # 18 des
        _, normal, diffuse, roughness, specular = \
            utils.read_image_material_tiled(os.path.join(self.gt_material_path, name))
        renderings, sm, hm, inl, inlr = utils.read_img_materials_auxrend(path, split=5)
        renderings = renderings[16:272, 16:272, :]
        normal = normal[16:272, 16:272, :]
        diffuse = diffuse[16:272, 16:272, :]
        roughness = roughness[16:272, 16:272, :]
        specular = specular[16:272, 16:272, :]
        inl = inl[16:272, 16:272, :]
        inlr = inlr[16:272, 16:272, :]

        materials = torch.FloatTensor(np.stack([normal, diffuse, roughness, specular], axis=0))
        materials = utils.adapt_roughness(materials, torch.device('cpu'))
        materials = utils.material_reshape(materials)  # [256, 256, 12]
        materials = utils.preprocess(materials)  # [0, 1] -> [-1, 1]

        renderings = utils.preprocess(renderings)
        inl = utils.preprocess(inl)
        inlr = utils.preprocess(inlr)
        
        return torch.FloatTensor(renderings), torch.FloatTensor(inl), torch.FloatTensor(inlr), materials, name.replace('.png', '')

class Des19DatasetLoad(torch.utils.data.Dataset):
    def __init__(self, gt_material_path, input_path, log_input):
        self.gt_material_path = gt_material_path
        self.input_path = input_path
        self.log_input = log_input

        self.load_path()

    def load_path(self):
        self.paths = glob.glob(self.input_path + "/*.png")
        self.dataset_len = len(self.paths)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        path = self.paths[index]
        name = path.split('/')[-1]

        # 19 des
        normal, diffuse, roughness, specular = \
            utils.read_img_materials_auxrend(os.path.join(self.gt_material_path, name), split=4) # 512*512
        renderings, sm, hm, inl, inlr = utils.read_img_materials_auxrend(path, split=5) # already 256*256
        normal = cv2.resize(normal, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        diffuse = cv2.resize(diffuse, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        roughness = cv2.resize(roughness, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        specular = cv2.resize(specular, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

        materials = torch.FloatTensor(np.stack([normal, diffuse, roughness, specular], axis=0))
        materials = utils.adapt_roughness(materials, torch.device('cpu'))
        materials = utils.material_reshape(materials)  # [256, 256, 12]
        materials = utils.preprocess(materials)  # [0, 1] -> [-1, 1]

        renderings = utils.preprocess(renderings)
        inl = utils.preprocess(inl)
        inlr = utils.preprocess(inlr)
        
        return torch.FloatTensor(renderings), torch.FloatTensor(inl), torch.FloatTensor(inlr), materials, name.replace('.png', '')


class RealDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.load_path()

    def load_path(self):
        self.paths = glob.glob(self.dataset_path + "/*.png")
        self.dataset_len = len(self.paths)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        path = self.paths[index]
        name = path.split('/')[-1]

        img = utils.read_material(path) #[0,1]
        img = utils.preprocess(img) #[-1,1]
        return torch.FloatTensor(img), name.replace('.png', '')