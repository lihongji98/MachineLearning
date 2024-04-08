import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from model import Transformer
from utils import (warmup_decay_learningrate, get_data_loader, generate_voc_buffer,
                   model_save, plot_train_test_epoch_loss, plot_train_test_iteration_loss)
from tqdm import tqdm


class EntityDataset(Dataset):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg
        self.MAX_LEN = 128

    def __len__(self):
        return len(self.src)

    def __getitem__(self, item):
        src = self.src[item]
        trg = self.trg[item]

        src = [1] + src + [2]
        trg = [1] + trg + [2]

        src_padding_len = self.MAX_LEN - len(src)
        trg_padding_len = self.MAX_LEN - len(trg)

        src = src + ([0] * src_padding_len)
        trg = trg + ([0] * trg_padding_len)

        src = torch.tensor(src, dtype=torch.long)
        trg = torch.tensor(trg, dtype=torch.long)

        return src, trg


if __name__ == "__main__":
    voc_en = generate_voc_buffer("en")
    voc_no = generate_voc_buffer("no")

    base_learning_rate, max_learning_rate, end_learning_rate = 1e-10, 1e-3, 1e-7
    batch_size = 256
    epochs = 10

    train_no, test_no, train_en, test_en = get_data_loader("./data/clean.no", "./data/clean.en", voc_no, voc_en)
    train_dataset = EntityDataset(train_no, train_en)
    test_dataset = EntityDataset(test_no, test_en)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(src_vocab_num=8000, trg_vocab_num=8000,
                        max_len=128,
                        embedding_dim=512, stack_num=3, ffn_dim=1024, qkv_dim=64, head_dim=8, device=device).to(device)

    optimizer = AdamW(model.parameters(), lr=base_learning_rate)

    iteration_per_epoch = len(train_data_loader)
    warmup_iteration = 1 * iteration_per_epoch
    end_iteration = epochs * iteration_per_epoch
    lr_scheduler = LambdaLR(optimizer,
                            lr_lambda=lambda current_iter: warmup_decay_learningrate(current_iter, warmup_iteration, end_iteration,
                                                                                     base_learning_rate, max_learning_rate, end_learning_rate))

    trg_pad_idx = voc_en.get("<pad>", 0)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.1)

    train_epoch_log_path = 'logs/train_epoch_log.txt'
    test_epoch_log_path = 'logs/test_epoch_log.txt'
    train_iteration_log_path = 'logs/train_iteration_log.txt'
    test_iteration_log_path = 'logs/test_iteration_log.txt'

    model.train()
    current_iteration = 0
    for epoch in range(epochs):
        model.train()
        train_losses, train_iteration_losses = [], []
        iteration_loss = 0
        loop = tqdm(enumerate(train_data_loader), desc="Training")
        for index, (source, target) in loop:
            source = source.to(device)
            target = target.to(device)

            output = model(source, target[:, :-1])
            output = output.reshape(-1, output.shape[2])
            target = target[:, 1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            current_iteration += 1
            iteration_loss += loss.item()

            train_avg_loss = iteration_loss / (index + 1)

            loop.set_postfix(percentage=str(np.round((index + 1) / len(train_data_loader) * 1e2, 2)) + "%",
                             epoch=str(epoch + 1) + "/" + str(epochs),
                             amount=len(train_data_loader),
                             loss="{:.2e}".format(loss.item()),
                             avg_loss="{:.2e}".format(train_avg_loss),
                             lr="{:.2e}".format(optimizer.param_groups[0]['lr']))

            with open(train_iteration_log_path, 'a') as train_iteration_log:
                train_iteration_log.write(f'iteration: {index+1}, loss: {loss.item()}\n')
            train_iteration_losses.append(loss.item())

        train_loss = iteration_loss / len(train_data_loader)
        train_losses.append(train_loss)
        with open(train_epoch_log_path, 'a') as train_epoch_log:
            train_epoch_log.write(f'epoch: {epoch}, loss: {train_loss}\n')

        model.eval()
        best_loss = torch.inf
        with torch.no_grad():
            test_losses, test_iteration_losses = [], []
            iteration_loss = 0
            loop = tqdm(enumerate(test_data_loader), desc="Testing")
            for index, (source, target) in loop:
                source = source.to(device)
                target = target.to(device)

                output = model(source, target[:, :-1])
                output = output.reshape(-1, output.shape[2])
                target = target[:, 1:].reshape(-1)

                loss = criterion(output, target)
                iteration_loss += loss.item()

                test_avg_loss = iteration_loss / (index + 1)

                loop.set_postfix(amount=len(test_data_loader),
                                 avg_loss="{:.2e}".format(test_avg_loss),
                                 loss="{:.2e}".format(loss.item()))

                with open(test_iteration_log_path, 'a') as test_iteration_log:
                    test_iteration_log.write(f'iteration: {index + 1}, loss: {loss.item()}\n')
                test_iteration_losses.append(loss.item())

            test_loss = iteration_loss / len(test_data_loader)
            test_losses.append(test_loss)
            with open(test_epoch_log_path, 'a') as test_epoch_log:
                test_epoch_log.write(f'epoch: {epoch}, loss: {test_loss}\n')

        plot_train_test_epoch_loss(train_losses, test_losses)
        plot_train_test_iteration_loss(train_iteration_losses)
        best_loss = model_save(model, optimizer, lr_scheduler, test_loss, best_loss, epoch, epochs)
