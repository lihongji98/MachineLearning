import pandas as pd
import torch
from torch.utils.data import Dataset


class EntityDataset(Dataset):
    def __init__(self, texts, tags, tokenizer):
        self.texts = texts
        self.tags = tags
        self.MAX_LEN = 128
        self.TOKENIZER = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        tags = self.tags[item]

        ids = []
        target_tag = []

        for i, s in enumerate(text):
            inputs = self.TOKENIZER.encode(str(s), add_special_tokens=False)

            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)

        ids = ids[:self.MAX_LEN - 2]
        target_tag = target_tag[:self.MAX_LEN - 2]

        ids = [101] + ids + [102]  # 101 -> CLS  102 -> SEP
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = self.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }


def process_data(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    df.loc[:, "Sentence #"] = df["Sentence #"].ffill()

    tag_values = sorted(set(df["Tag"].values))
    id_to_tag_dict = {k: v for k, v in enumerate(tag_values)}

    tag_to_id_dict = {v: k for k, v in id_to_tag_dict.items()}

    df["Tag"] = df["Tag"].apply(lambda x: tag_to_id_dict.get(x, None))

    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values

    return sentences, tag, id_to_tag_dict


def warmup_decay_learningrate(current_iteration, warmup_iteration, end_iteration, base_lr, max_lr, end_lr, x_dim,
                              lr_decay_strategy):
    """
    :param lr_decay_strategy: "linear_decay" or "noam_decay"
    """
    if lr_decay_strategy == "linear_decay":
        if current_iteration < warmup_iteration:
            slope = (max_lr - base_lr) / warmup_iteration
            current_lr = base_lr + slope * current_iteration
        else:
            slope = (end_lr - max_lr) / (end_iteration - warmup_iteration)
            current_lr = max_lr + slope * (current_iteration - warmup_iteration)
        return current_lr / base_lr

    elif lr_decay_strategy == "noam_decay":
        current_lr = 1 / torch.sqrt(torch.tensor(x_dim)) * torch.minimum(
            1 / torch.sqrt(torch.tensor(current_iteration)),
            torch.tensor(current_iteration / warmup_iteration ** 1.5))
        return current_lr / base_lr

    else:
        print("input correct learning rate decay: ['linear_decay', 'noam_decay']")
