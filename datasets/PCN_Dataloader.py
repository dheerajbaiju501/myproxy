import sys
from time import sleep
import pdb
sys.path.append(".")

import os
import random

import torch
import torch.utils.data as data
import numpy as np
import open3d as o3d
from torch.utils.data import DataLoader


id2cat = {
            # seen categories
            "Chinese": "Chinese",
            "Dutch": "Dutch",
            "Hungarian": "Hungarian",
            "SNCF": "SNCF"
            
        }

class ShapeNet(data.Dataset):
    """
    ShapeNet dataset in "PCN: Point Completion Network". It contains 28974 training
    samples while each complete samples corresponds to 8 viewpoint partial scans, 800
    validation samples and 1200 testing samples.
    """

    def __init__(self, dataroot, split, category):
        assert split in ["train", "valid", "test", "test_novel"], "split error value!"

        self.cat2id = {
            # seen categories
            "Chinese": "Chinese",
            "Dutch": "Dutch",
            "Hungarian": "Hungarian",
            "SNCF": "SNCF"
            
        }

        # self.id2cat = {cat_id: cat for cat, cat_id in self.cat2id.items()}

        self.dataroot = dataroot
        self.split = split
        self.category = category

        self.partial_paths, self.complete_paths = self._load_data()

    def __getitem__(self, index):
        if self.split == "train":
            partial_path = self.partial_paths[index].format(random.randint(1, 210))
        else:
            partial_path = self.partial_paths[index]

        complete_path = self.complete_paths[index]

        # print("p: ", partial_path)
        # print("c: ", complete_path)

        #partial_pc = self.random_sample(self.read_point_cloud(partial_path), 2048)
        partial_pc = self.read_point_cloud(partial_path)
        complete_pc = self.read_point_cloud(complete_path)
        #complete_pc = self.random_sample(self.read_point_cloud(complete_path), 16384)
        # pdb.set_trace()
        # print(self.partial_paths[index])
        # print(self.partial_paths[index].split('/')[6])
        # exit(0)
        category_name = id2cat[str(self.partial_paths[index].split('/')[6])]
        a1 = self.read_point_cloud(partial_path)
        # print("partial: ", a1.shape)
        a2 = self.read_point_cloud(complete_path)
        # print("complete: ", a2.shape)
        a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
        a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])
        # print(original_c.shape)
        # print(original_p.shape)
        missing_part = np.setdiff1d(a2_rows, a1_rows).view(a2.dtype).reshape(-1, a2.shape[1])
        # print("missing_part: ", missing_part.shape)
        # sleep(10)

        return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc), category_name

    def __len__(self):
        return len(self.complete_paths)

    def _load_data(self):
        with open(os.path.join(self.dataroot, "{}.list").format(self.split), "r") as f:
            lines = f.read().splitlines()

        if self.category != "all":
            lines = list(
                filter(lambda x: x.startswith(self.cat2id[self.category]), lines)
            )

        partial_paths, complete_paths = list(), list()

        for line in lines:
            category, model_id = line.split("/")
            if self.split == "train":
                partial_paths.append(
                    os.path.join(
                        self.dataroot,
                        self.split,
                        "partial",
                        category,
                        model_id + "_{}.ply",
                    )
                )
            else:
                partial_paths.append(
                    os.path.join(
                        self.dataroot,
                        self.split,
                        "partial",
                        category,
                        model_id + ".ply",
                    )
                )
            complete_paths.append(
                os.path.join(
                    self.dataroot, self.split, "complete", category, model_id + ".ply"
                )
            )

        return partial_paths, complete_paths

    def read_point_cloud(self, path):
        pc = o3d.io.read_point_cloud(path)
        return np.array(pc.points, np.float32)

    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate(
                [idx, np.random.randint(pc.shape[0], size=n - pc.shape[0])]
            )
        return pc[idx[:n]]


if __name__ == '__main__':
    val_dataset = ShapeNet("data/My_PCN/PCN", "train", "all")
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=8,
    )

    for p, c, cat_name in val_dataloader:
        # print(p.size())
        # print(c.size())
        # print(cat_name)
        break
