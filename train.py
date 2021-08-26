from utils import AverageMeter
import numpy as np
import os
import argparse
import logging
from tqdm import tqdm
import shutil

from models.gfnet import GFNet
from models.res2net import res2net50_26w_4s_bi
from roc_plot import val_last
from transfer import transfer_loader, WarmupLinearSchedule

import torch
import torch.nn as nn

from data import get_dataloader
from utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class Net_Train():
    def __init__(self, args) -> None:
        self.args = args
        
        self.task_name = args.task_name
        self.lr = args.lr
        self.epochs = args.epochs
        self.warmup_epochs = args.warmup_epochs
        self.batch_size = args.batch_size
        self.device = args.device
        self.data_path = args.data_path

        self.lr_decay_step = args.lr_decay_step
        self.lr_decay_gamma = args.lr_decay_gamma
        self.eval_batch_size = args.eval_batch_size
        self.eval_period = args.eval_period
        self.log_save_dir = args.log_save_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.num_workers = args.num_workers
        self.transfer = args.transfer
        self.writer = SummaryWriter(self.log_save_dir + '/' + self.task_name + '_logs', comment=args.model)

        if args.model == 'gfnet':
            self.model = GFNet(num_classes=9, drop_rate=args.drop_rate)
        else:
            self.model = res2net50_26w_4s_bi(pretrained=False, num_classes=9)
        self.set_up()

    def get_lr_scheduler(self, optimizer, step_size=40, gamma=0.7, last_epoch=-1):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma, last_epoch)
        return scheduler

    def set_up(self):
        if not os.path.exists(self.log_save_dir):
            os.makedirs(self.log_save_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_dir = self.checkpoint_dir + '/' + self.task_name
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.log_file_path = self.log_save_dir+'/'+self.task_name+".txt"
        with open(self.log_file_path, 'w') as log_file:
            log_file.write("train_loss,train_acc,test_loss,test_acc\n")

    def train(self):
        
        train_dl, test_dl = get_dataloader(self.data_path, self.batch_size, self.eval_batch_size, self.device, self.num_workers)
        print('\n', self.args, '\n')

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr)

        if self.transfer:
            self.model = transfer_loader(self.model, self.transfer)
            self.lr_scheduler = WarmupLinearSchedule(optimizer, self.warmup_epochs, self.epochs)
            logger.info("***** Start transfer learning *****")
        else:
            self.lr_scheduler = self.get_lr_scheduler(optimizer, self.lr_decay_step, self.lr_decay_gamma)

        loss_func = nn.CrossEntropyLoss()
        train_loss_meter = AverageMeter()
        
        self.model.to(self.device)
        self.model.train()
        logger.info("***** Start training *****")

        self.current_lr = self.lr
        self.best_acc = 0
        for epoch in range(1, self.epochs + 1):
            train_batch_iter = tqdm(train_dl,
                        desc="Training (X / X epochs) (loss=X.X)",
                        bar_format="{l_bar}{r_bar}",
                        dynamic_ncols=True)
            self.current_epoch = epoch
            for x, y in train_batch_iter:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = loss_func(pred, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss_meter.update(loss.item())
                train_batch_iter.set_description(
                    "Training (%d / %d epochs) (loss=%2.5f)" % (epoch, self.epochs, train_loss_meter.val)
                )   

            self.lr_scheduler.step()           

            if epoch % self.eval_period == 0 and epoch >= 0:
                self.model_eval(test_dl, train_dl)
                if self.best_acc < self.val_accuracy:
                    self.best_acc = self.val_accuracy
                    self.writer.add_scalar('best_acc', self.best_acc, self.current_epoch)

                    if os.path.exists(self.checkpoint_dir):
                        shutil.rmtree(self.checkpoint_dir)
                        os.makedirs(self.checkpoint_dir)
                    checkpoint_save_path = self.checkpoint_dir + '/' + \
                                            self.task_name + "_ckpt.pth"
                    torch.save(self.model.state_dict(), checkpoint_save_path)
                    logger.info(f"Saving checkpoint to {checkpoint_save_path}")

            if epoch >= self.eval_period:
                with open(self.log_file_path, 'a') as log_file:
                    log_file.write(f"{self.train_loss_meter.avg},{self.train_accuracy},{self.eval_loss_meter.avg},{self.val_accuracy}\n")
        
        val_last(test_dl, self.model, nn.CrossEntropyLoss, optimizer, self.args.model)
        logger.info("Best Accuracy: \t%f" % self.best_acc)
        logger.info("End Training!")

    def model_eval(self, test_loader, train_loader):

        self.eval_loss_meter = AverageMeter()
        self.train_loss_meter = AverageMeter()

        logger.info("***** Running Validation *****")
        logger.info("  Batches of testset: %d", len(test_loader))
        logger.info("  Batch size: %d", self.eval_batch_size)

        self.model.eval()
        all_preds, all_labels = [], []
        test_iter = tqdm(test_loader,
                    desc="Validating... (loss=X.X)",
                    dynamic_ncols=True)
        train_iter = tqdm(train_loader,
            desc="Validating... (loss=X.X)",
            dynamic_ncols=True)

        eval_loss_func = nn.CrossEntropyLoss()
        with torch.no_grad():
            for x, y in test_iter:
                x = x.to(self.device)
                y = y.to(self.device)
                # (N, 100, 75)
                pred = self.model(x)
                eval_loss = eval_loss_func(pred, y)
                self.eval_loss_meter.update(eval_loss.item())

                preds = torch.argmax(pred, dim=-1) # -> (N,)
                all_preds.append(preds)
                all_labels.append(y)

                test_iter.set_description("Validating... (loss=%2.5f)" % self.eval_loss_meter.val)

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)


            self.val_accuracy = (all_preds == all_labels).detach().cpu().numpy().mean()

            all_preds, all_labels = [], []
            for x, y in train_iter:
                x = x.to(self.device)
                y = y.to(self.device)
                # (N, 100, 75)
                pred = self.model(x)
                train_loss = eval_loss_func(pred, y)
                self.train_loss_meter.update(train_loss.item())

                preds = torch.argmax(pred, dim=-1) # -> (N,)
                all_preds.append(preds)
                all_labels.append(y)

                train_iter.set_description("Validating train... (loss=%2.5f)" % self.train_loss_meter.val)

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            self.train_accuracy = (all_preds == all_labels).detach().cpu().numpy().mean()

        self.current_lr = self.lr_scheduler.get_last_lr()[0]

        self.writer.add_scalar('train_loss', self.train_loss_meter.avg, self.current_epoch)
        self.writer.add_scalar('train_acc', self.train_accuracy, self.current_epoch)
        self.writer.add_scalar('val_loss', self.eval_loss_meter.avg, self.current_epoch)
        self.writer.add_scalar('val_acc', self.val_accuracy, self.current_epoch)


        self.model.train()
        logger.info("\n")
        logger.info("Validation Results")
        logger.info("Current Learning Rate: %2.5f" % self.current_lr)
        logger.info("Valid Loss: %2.5f" % self.eval_loss_meter.avg)
        logger.info("Valid Accuracy: %2.5f" % self.val_accuracy)
        logger.info("Train Loss: %2.5f" % self.train_loss_meter.avg)
        logger.info("Train Accuracy: %2.5f" % self.train_accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--task_name", default="task0", type=str)
    parser.add_argument("--model", default="gfnet", choices=["gfnet", "res2net"], type=str)
    parser.add_argument("--data_path", default="./data/NEU_dataset", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--warmup_epochs", default=3, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=8, type=int)   # for tesla v100 512
    parser.add_argument("--eval_batch_size", default=8, type=int) # for tesla v100 512
    parser.add_argument("--eval_period", default=1, type=int)
    parser.add_argument("--lr_decay_step", default=5, type=int)
    parser.add_argument("--lr_decay_gamma", default=0.7, type=float)
    parser.add_argument("--drop_rate", default=0.5, type=float)
    parser.add_argument("--log_save_dir", default="./log", type=str)
    parser.add_argument("--checkpoint_dir", default="./checkpoint", type=str)
    parser.add_argument("--num_workers", default=0, type=int) # num_workers for linux to load dataset, 0 for windows
    parser.add_argument("--transfer", default="", type=str) # ./checkpoint/task0/task0_ckpt.pth

    args = parser.parse_args()
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
    net = Net_Train(args)
    net.train()