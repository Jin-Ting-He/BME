# torchrun --nproc_per_node=2 --master_port=29501 home/jthe/blur_detection/blur_detector/main_ddp.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import cv2
import random
import argparse
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# set random seed
torch.manual_seed(39)
torch.cuda.manual_seed(39)
random.seed(39)
np.random.seed(39)
from torch.utils.data import DataLoader
from dataset.dataloader import BlurMagDataset
from utils.logger import Logger
from model.bme_model import MyNet_Res50_multiscale
from train.optimizer import Optimizer

class Trainer():
    def __init__(self, args) -> None:
        self.args = args
        if not self.args.test_only:
            self.model_setting()
            self.dataset_setting()
            self.weight_path = self.args.weight_path
            os.makedirs(self.weight_path, exist_ok=True)

            if dist.get_rank() == 0:
                self.logger = Logger(self.args.logger_path)

            self.l1_loss = torch.nn.L1Loss()
            self.opt = Optimizer(self.model, self.args)

            self.min_loss = float('inf')

    def train(self):
        for epoch in range(self.args.epochs):
            self.model.train()

            train_loss = self.train_one_epoch(epoch)

            test_loss = self.test(epoch)

            if dist.get_rank() == 0:
                self.logger.register(epoch, train_loss, test_loss)

                if (test_loss < self.min_loss):
                    self.min_loss = test_loss
                    torch.save(self.model.module.state_dict(), os.path.join(self.weight_path, "best_net.pth"))
                    self.logger.save_best(epoch)
                if (epoch+1) % 10 == 0:
                    torch.save(self.model.module.state_dict(), os.path.join(self.weight_path, f"epoch_{epoch+1}.pth"))

    def train_one_epoch(self, epoch):
        total_loss = 0
        num_samples = 0
        self.train_sampler.set_epoch(epoch)
        pbar = tqdm(self.train_dl, total=len(self.train_dl))
        pbar.set_description(f'Epoch [{epoch}/{self.args.epochs}] training')
        for i,data in enumerate(pbar):
            blur_img, blur_mag = data
            blur_img, blur_mag = blur_img.to(self.args.device),blur_mag.to(self.args.device)

            pred_mag = self.model(blur_img)

            loss = self.l1_loss(pred_mag, blur_mag) + self.sill_loss(pred_mag, blur_mag)

            loss.backward()
            self.opt.step()
            self.opt.zero_grad() 

            total_loss += loss.detach().item()
            num_samples += 1
            avg_loss = total_loss / num_samples

            pbar.set_description(f"Epoch {epoch+1} Loss: {avg_loss:.4f} Lr: {self.opt.get_lr()}")
            pbar.update(1)
        pbar.close()    
        self.opt.lr_schedule()
        return avg_loss

    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            num_samples = 0
            self.test_sampler.set_epoch(epoch)
            pbar = tqdm(self.test_dl, total=len(self.test_dl))
            pbar.set_description(f'Epoch [{epoch}/{self.args.epochs}] testing')
            for i,data in enumerate(pbar):
                blur_img, blur_mag = data
                blur_img, blur_mag = blur_img.to(self.args.device),blur_mag.to(self.args.device)
                
                pred_mag = self.model(blur_img)
                
                pred_mag = pred_mag.squeeze(1)
                loss = self.l1_loss(pred_mag, blur_mag)

                total_loss += loss.detach().item()
                num_samples += 1
                avg_loss = total_loss / num_samples

                pbar.set_description(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
                pbar.update(1)
            pbar.close() 
            return avg_loss

    def inference(self):
        # load model
        checkpoint_path = os.path.join(self.args.weight_path, self.args.model_name)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
        self.model = MyNet_Res50_multiscale().cuda()
        self.model.load_state_dict(checkpoint)
        print("Loading Model Done")

        # load dataset
        infer_dataset = BlurMagDataset(dataset_root=self.args.infer_dataset_path,train=False)
        infer_dl = DataLoader(infer_dataset, batch_size=1, shuffle=False, pin_memory=True)
        print("Loading Dataset Done")

        output_folder = self.args.infer_output_path
        os.makedirs(output_folder, exist_ok=True)

        print("Starting Evaluation")
        self.model.eval()
        with torch.no_grad():
            for i,data in enumerate(infer_dl):
                print("Complete: ", i)
                blur_img, video_name, file_name = data
                blur_img = blur_img.cuda()

                output = self.model(blur_img)
                # output = output.clamp(-0.5, 0.5)
                output = output.clamp(0 ,1)
                output = output[0].to('cpu').detach().numpy().squeeze()
                # output =  ((output+0.5) * 205)
                output =  ((output) * 205)
                output = output/np.max(output)
                output = np.uint8(255-(output*255))

                output_video_folder = os.path.join(output_folder, video_name[0])
                os.makedirs(output_video_folder, exist_ok=True)
                output_path = os.path.join(output_video_folder,file_name[0])
                cv2.imwrite(output_path, output)
                # output.save(output_path)

    def model_setting(self):
        if self.args.local_rank != -1:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
            dist.init_process_group(backend="nccl", init_method='env://')
        self.args.device = device
        print("device:", self.args.device)
        num_gpus = torch.cuda.device_count()
        print("# of gpus",num_gpus)

        self.model = MyNet_Res50_multiscale()
        self.model.to(self.args.device)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank)
 
    def dataset_setting(self):
        dataset_train = BlurMagDataset(dataset_root=self.args.training_dataset_path,train=True)
        self.train_sampler = DistributedSampler(dataset_train)
        self.train_dl = DataLoader(dataset_train, sampler=self.train_sampler, batch_size=self.args.batch_size,pin_memory=True,
                drop_last=True,num_workers=4)
        
        dataset_test = BlurMagDataset(dataset_root=self.args.testing_dataset_path,train=True)
        self.test_sampler = DistributedSampler(dataset_test)
        self.test_dl = DataLoader(dataset_test, sampler=self.test_sampler, batch_size=self.args.batch_size,pin_memory=True,
                drop_last=True,num_workers=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--weight_path", default="home/jthe/BME/BME/weights/", type=str)
    parser.add_argument("--logger_path", default="home/jthe/BME/BME/log/myunet_v3/training_loss.txt", type=str)
    parser.add_argument("--training_dataset_path", default="disk2/jthe/datasets/GOPRO_blur_magnitude/train", type=str)
    parser.add_argument("--testing_dataset_path", default="disk2/jthe/datasets/GOPRO_blur_magnitude/test", type=str)
    parser.add_argument("--infer_dataset_path", default="disk2/jthe/datasets/GOPRO_blur_magnitude/test/frame11", type=str)
    parser.add_argument("--infer_output_path", default="home/jthe/BME/BME/output/", type=str)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--init_lr", default=1e-3, type=float)
    parser.add_argument("--final_lr", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument("--model_name", default="best_net.pth", type=str)
    args = parser.parse_args()

    trainer = Trainer(args)
    if args.test_only:
        trainer.inference()
    else:
        trainer.train()