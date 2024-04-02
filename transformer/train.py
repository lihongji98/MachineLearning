import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model import Transformer

from collections import defaultdict
from itertools import islice

from sklearn.model_selection import train_test_split

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


def get_data_loader(src_path, trg_path, src_voc, trg_voc):
    src_ids = []
    trg_ids = []

    with open(src_path, 'r', encoding='utf-8') as src:
        data = src.readlines()
        for tokens in tqdm(data):
            tokens = tokens.split(" ")[:-1]
            token_ids = [src_voc.get(tokens[i], src_voc["<unk>"]) for i in range(len(tokens))]
            src_ids.append(token_ids)

    with open(trg_path, 'r', encoding='utf-8') as trg:
        data = trg.readlines()
        for tokens in tqdm(data):
            tokens = tokens.split(" ")[:-1]
            token_ids = [trg_voc.get(tokens[i], trg_voc["<unk>"]) for i in range(len(tokens))]
            trg_ids.append(token_ids)

    src_train, src_test, trg_train, trg_test = train_test_split(src_ids, trg_ids, test_size=0.15, random_state=42)

    return src_train, src_test, trg_train, trg_test


def generate_voc_buffer(lang):
    vocab_table = defaultdict(lambda: len(vocab_table))
    vocab_table.default_factory = vocab_table.__len__

    with open(f'./model/voc_{lang}.txt', 'r', encoding='utf-8') as f:
        for line in islice(f, 8000):  # Read only the first 8000 lines
            token, index_str = line.strip().split()
            vocab_table[token] = int(index_str)

    return vocab_table


if __name__ == "__main__":
    voc_en = generate_voc_buffer("en")
    voc_no = generate_voc_buffer("no")

    learning_rate = 3e-4
    batch_size = 128
    epochs = 3

    train_no, test_no, train_en, test_en = get_data_loader("./data/clean.no", "./data/clean.en", voc_no, voc_en)
    train_dataset = EntityDataset(train_no, train_en)
    test_dataset = EntityDataset(test_no, test_en)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(src_vocab_num=8000, trg_vocab_num=8000,
                        max_len=128,
                        embedding_dim=256, stack_num=3, ffn_dim=1024, qkv_dim=64, head_dim=4, device=device).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    trg_pad_idx = voc_en.get("<pad>", 0)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.1)

    for epoch in range(epochs):
        print("Epoch {}/{}:".format(epoch + 1, epochs))

        model.train()
        train_losses = []
        iteration_loss = 0
        for source, target in tqdm(train_data_loader):
            source = source.to(device)
            target = target.to(device)

            output = model(source, target[:, :-1])
            output = output.reshape(-1, output.shape[2])

            target = target[:, 1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            iteration_loss += loss.item()

        train_loss = iteration_loss / len(train_data_loader)
        train_losses.append(train_loss)

        print(train_losses[-1])
