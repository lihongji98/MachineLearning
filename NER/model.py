import numpy as np
import torch.nn as nn
import transformers
from TorchCRF import CRF
from tqdm import tqdm


class NER_Recognizer(nn.Module):
    def __init__(self, num_tag, bert_path='bert-base-uncased'):
        super().__init__()
        self.num_tag = num_tag
        self.bert = transformers.BertModel.from_pretrained(bert_path)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(bert_path, do_lower_case=True)
        self.fc = nn.Linear(768, self.num_tag)
        self.lstm = nn.LSTM(768, 768 // 2, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.crf = CRF(self.num_tag, batch_first=True)

    def forward(self, ids, mask, token_type_ids, target_tag):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        seq_out, _ = self.lstm(o1)
        tag = self.fc(seq_out)
        crf_tag = self.crf.decode(tag, mask.bool())

        return crf_tag

    def Loss_fn(self, ids, mask, token_type_ids, target_tag):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        seq_out, _ = self.lstm(o1)
        y_pred = self.fc(seq_out)
        loss = -self.crf.forward(y_pred, target_tag, mask.bool(), reduction='mean')

        return loss


def train_fn(data_loader, model, optimizer, device, scheduler, epoch):
    model.train()
    train_iteration_loss = 0
    loop = tqdm(enumerate(data_loader), desc="Training")
    for index, data in loop:
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        loss = model.Loss_fn(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_iteration_loss += loss.item()

        train_avg_loss = train_iteration_loss / (index + 1)

        loop.set_postfix(percentage=str(np.round((index + 1) / len(data_loader) * 1e2, 2)) + "%",
                         epoch=str(epoch + 1),
                         amount=len(data_loader),
                         loss="{:.6e}".format(loss.item()),
                         avg_loss="{:.6e}".format(train_avg_loss),
                         lr="{:.2e}".format(optimizer.param_groups[0]['lr']))

    return train_iteration_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        loss = model.Loss_fn(**data)
        final_loss += loss.item()

    return final_loss / len(data_loader)
