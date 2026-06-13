
from config import Config
from abc import ABC, abstractclassmethod
import torch
class DataLoader:
    def __init__(self, config):
        self.path = config.path
        self.config = config
    @abstractclassmethod
    def get_batch(self):
        pass

    def read_file(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
        # support more formats in future

    def establish_vocabulary(self, text):
        chars = sorted(list(set(text)))
        stoi = { ch: i for i, ch in enumerate(chars)}
        itos = { i: ch for i, ch in enumerate(chars)}

        self.config.vocab_size = len(chars)
        self.config.encode = lambda x: [stoi[ch] for ch in x]  
        self.config.decode = lambda x: ''.join([itos[i] for i in x])

    

    def get_data_splits(self, text):
        data = torch.tensor(self.config.encode(text), dtype = torch.long)
        n = int(0.9*len(data))
        train = data[:n]
        val = data[n:]
        return train, val