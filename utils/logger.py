import os
import shutil
from datetime import datetime
from os.path import dirname, join

import torch


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)  # 创建日志文件目录
        self.logs = []

    def register(self, epoch, train_loss, test_loss):
        self.logs.append((epoch, train_loss, test_loss))
        with open(self.log_file, 'a') as f:
            f.write(f'Epoch {epoch}: Train Loss {train_loss} Test Loss {test_loss}\n')
            print(f'Epoch {epoch}: Train Loss {train_loss} Test Loss {test_loss}\n')
    def report(self):
        print(f'Logging results to {self.log_file}')
        for epoch, loss in self.logs:
            print(f'Epoch {epoch}: Loss {loss}')
    def save_best(self, epoch):
        with open(self.log_file, 'a') as f:
            f.write(f'Epoch {epoch}: Saving best model\n')