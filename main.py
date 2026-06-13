
from config import Config
from model import picoGpt
from data_loader.shakespeare_loader import SL
from trainer import Trainer
import torch

def main():

    config = Config(
        n_embed = 128,
        head_size = 16,
        num_heads = 8,
        n_layer = 3,
        vocab_size = 65,
        block_size = 16,
        lr = 1e-3 ,

        eval_interval = 100,
        device = 'mps' if torch.mps.is_available() else 'cpu',
        batch_size = 32,

        path = "./data/input.txt"
    )

    model = picoGpt()
    model.to(config.device) 
    print(sum(p.numel() for p in model.parameters())/1e6, 'M params')
    data_loader = SL(config)
    trainer = Trainer(config, model, data_loader)
    trainer.train()


main()
