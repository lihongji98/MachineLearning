from collections import defaultdict
from itertools import islice

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


class PairDataset(Dataset):
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


def warmup_decay_learningrate(current_iteration, warmup_iteration, end_iteration, base_lr, max_lr, end_lr):
    if current_iteration < warmup_iteration:
        slope = (max_lr - base_lr) / warmup_iteration
        current_lr = base_lr + slope * current_iteration
    else:
        slope = (end_lr - max_lr) / (end_iteration - warmup_iteration)
        current_lr = max_lr + slope * (current_iteration - warmup_iteration)

    return current_lr / base_lr


def get_data_loader(src_path, trg_path, src_voc, trg_voc):
    src_ids = []
    trg_ids = []

    with open(src_path, 'r', encoding='utf-8') as src:
        data = src.readlines()
        for tokens in data:
            tokens = tokens.split(" ")[:-1]
            token_ids = [src_voc.get(tokens[i], src_voc["<unk>"]) for i in range(len(tokens))]
            src_ids.append(token_ids)
    print("Source DataLoader is ready...")

    with open(trg_path, 'r', encoding='utf-8') as trg:
        data = trg.readlines()
        for tokens in data:
            tokens = tokens.split(" ")[:-1]
            token_ids = [trg_voc.get(tokens[i], trg_voc["<unk>"]) for i in range(len(tokens))]
            trg_ids.append(token_ids)
    print("Target DataLoader is ready...")

    src_train, src_test, trg_train, trg_test = train_test_split(src_ids, trg_ids, test_size=0.3, random_state=42)

    return src_train, src_test, trg_train, trg_test


def generate_voc_buffer(lang):
    vocab_table = defaultdict(lambda: len(vocab_table))
    vocab_table.default_factory = vocab_table.__len__

    with open(f'voc/voc_{lang}.txt', 'r', encoding='utf-8') as f:
        for line in islice(f, 7000):
            token, index_str = line.strip().split()
            vocab_table[token] = int(index_str)

    return vocab_table


def model_save(model, optimizer, lr_scheduler, test_loss, best_loss, epoch, epochs, save_epoch_num):
    if test_loss < best_loss:
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict()},
                   './parameters/best_checkpoint.pth')
        best_loss = test_loss
        tqdm.write(f"best loss model is saved at epoch {epoch}, current best loss: {best_loss}...")

    if epochs - save_epoch_num <= epoch < epochs:
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict()},
                   f'./parameters/checkpoint_{epoch}.pth')

    return best_loss


def plot_train_test_epoch_loss(train_losses, test_losses):
    plt.plot([i for i in range(len(train_losses))], train_losses, label='Training Loss')
    plt.plot([i for i in range(len(test_losses))], test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('epoch loss')
    plt.legend()
    plt.savefig('logs/epoch_loss.png')
    plt.close()


def plot_train_test_iteration_loss(train_iteration_losses):
    plt.plot([i for i in range(len(train_iteration_losses))], train_iteration_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('iteration loss')
    plt.savefig('logs/training_iteration_loss.png')
    plt.close()
