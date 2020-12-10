import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import distdl

num_workers = 0

class DummyLoader:

    def __init__(self, batch_size, n_data_points):

        self.batch_size = batch_size
        self.n_data_points = n_data_points

        self.n_loaded = 0

        self.mod_batch_size = n_data_points % batch_size
        self.n_batches = n_data_points // batch_size

    def __iter__(self):
        for i in range(self.n_batches):
            yield distdl.utilities.torch.zero_volume_tensor(self.batch_size), torch.zeros(self.batch_size)
        yield distdl.utilities.torch.zero_volume_tensor(self.mod_batch_size), torch.zeros(self.mod_batch_size)

def get_data_loaders(batch_size, train_data_dir, test_data_dir,
                     download=False, dummy=False):

    data_train = datasets.ImageFolder(train_data_dir,
                                      transforms.Compose([
                                          transforms.Resize((500, 1000)),
                                          transforms.ToTensor(),
                                      ]))

    data_test = datasets.ImageFolder(test_data_dir,
                                     transforms.Compose([
                                         transforms.Resize((500, 1000)),
                                         transforms.ToTensor(),
                                     ]))

    if not dummy:
        train_loader = DataLoader(data_train,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)
        test_loader = DataLoader(data_test,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
    else:
        train_loader = DummyLoader(batch_size, 14034)
        test_loader = DummyLoader(batch_size, 3000)

    return train_loader, test_loader
