
from .data_loader import DataLoader
import torch

class SL(DataLoader):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        text = self.read_file()
        self.establish_vocabulary(text)
        self.train, self.val = self.get_data_splits(text)

    def get_batch(self, split):
        data =  self.train if split == 'train' else self.val
        ix = torch.randint(len(data)- self.config.block_size, (self.config.batch_size,))
        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1: i+self.config.block_size+1] for i in ix])
        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y