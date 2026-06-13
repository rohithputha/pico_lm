import torch



class Trainer:
    def __init__(self, config, model, data_loader):
        self.config = config
        self.model = model
        self.data_loader = data_loader

    def train(self):
        lr = 1e-3
        optim = torch.optim.AdamW(self.model.parameters(), lr = lr )
        max_iters = 5000 # these should be configurable
        


        @torch.no_grad()
        def estimate_loss():
            out = {}
            self.model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(200)
                for k in range(200):
                    X, Y = self.data_loader.get_batch(split)
                    logits , loss = self.model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            self.model.train()

            return out

        for iter in range(max_iters):
            if iter % self.config.eval_interval == 0 or iter == max_iters -1:
                losses = estimate_loss()
                print(f"step{iter}: train loss: {losses['train']},  val loss: {losses['val']}")
            
            xb, yb = self.data_loader.get_batch('train')
            logits, loss = self.model(xb,yb)
            optim.zero_grad(set_to_none= True)
            loss.backward()
            optim.step()