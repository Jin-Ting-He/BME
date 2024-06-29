from importlib import import_module

import torch.optim.lr_scheduler as lr_scheduler

class Optimizer:
    def __init__(self, model, args):
        # create optimizer
        # trainable = filter(lambda x: x.requires_grad, model.parameters())
        trainable = model.parameters()
        optimizer_name = 'Adam'
        lr = args.init_lr
        module = import_module('torch.optim')
        self.optimizer = getattr(module, optimizer_name)(trainable, lr=lr)
        # create scheduler
        T_max = args.epochs  # 這裡的 T_max 是 Cosine Annealing 調度器中的一個參數，代表了學習率周期的迭代次數
        eta_min = args.final_lr  # eta_min 是學習率下降到的最小值，在 Cosine Annealing 中，學習率會在 eta_min 和初始學習率之間變化
        
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
            

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_schedule(self):
        self.scheduler.step()