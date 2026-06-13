from model import model, get_batch, eval_interval

import torch

lr = 1e-3
optim = torch.optim.AdamW(model.parameters(), lr = lr )
max_iters = 5000


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            logits , loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters -1:
        losses = estimate_loss()
        print(f"step{iter}: train loss: {losses['train']},  val loss: {losses['val']}")
    
    xb, yb = get_batch('train')
    logits, loss = model(xb,yb)
    optim.zero_grad(set_to_none= True)
    loss.backward()
    optim.step()