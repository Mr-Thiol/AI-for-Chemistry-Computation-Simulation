import os
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from utils import load_arc

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


class PBCDataset(InMemoryDataset):
    def __init__(self, root: str, cutoff=5.0, transform=None, pre_transform=None, pre_filter=None):
        self.cutoff = cutoff
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed")


    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        _list = load_arc.load(self.raw_dir, cutoff=self.cutoff)
        data_list = []
        for data in _list:
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


def dataloader(BATCH_SIZE, percent,cutoff):
    path = os.path.join(FILE_PATH, "data", "train")
    dataset = PBCDataset(path,cutoff)
    train_dataset, valid_dataset = random_split(dataset, percent)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    info = {
        "train_count": len(train_dataset),
        "valid_count": len(valid_dataset),
    }
    print(info)
    return train_loader, valid_loader, info


if __name__ == '__main__':
    dataloader(512,[0.95,0.05],5.0)
