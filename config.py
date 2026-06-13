
from dataclasses import dataclass

@dataclass
class Config:
    device: str
    n_embed: int
    head_size:int
    num_heads:int
    n_layer:int
    vocab_size:int
    block_size:int
    lr: float
    eval_interval: int
    batch_size: int
    
    path: str


