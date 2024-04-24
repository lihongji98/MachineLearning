import torch
import transformers
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from model import NER_Recognizer, train_fn, eval_fn
from NER.utils import process_data, EntityDataset
from utils import warmup_decay_learningrate

if __name__ == "__main__":
    batch_size = 96
    epochs = 10
    device = torch.device("cuda")

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    sentences, tag, id_to_tag_dict = process_data("data/data.csv")

    train_sentences, teva_sentences, train_tag, teva_tag = train_test_split(sentences, tag, random_state=42, test_size=0.1)
    test_sentences, valid_sentences, test_tag, valid_tag = train_test_split(teva_sentences, teva_tag, random_state=42, test_size=0.5)

    train_dataset = EntityDataset(texts=train_sentences, tags=train_tag, tokenizer=tokenizer)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = EntityDataset(texts=valid_sentences, tags=valid_tag, tokenizer=tokenizer)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size)

    num_tag = len(id_to_tag_dict.keys())
    model = NER_Recognizer(num_tag=num_tag).to(device)

    iteration_per_epoch = len(train_data_loader)
    end_iteration = epochs * iteration_per_epoch
    optimizer = AdamW(model.parameters(), lr=1e-10)
    lr_scheduler = LambdaLR(optimizer,
                            lr_lambda=lambda current_iter: warmup_decay_learningrate(current_iter,
                                                                                     warmup_iteration=50,
                                                                                     end_iteration=end_iteration,
                                                                                     base_lr=1e-10,
                                                                                     max_lr=3e-4,
                                                                                     end_lr=1e-7,
                                                                                     x_dim=768,
                                                                                     lr_decay_strategy="noam_decay"))

    best_loss = torch.inf
    for epoch in range(epochs):
        train_loss = train_fn(train_data_loader, model, optimizer, device, lr_scheduler, epoch)
        test_loss = eval_fn(valid_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), "parameter/ner_bert_lstm_crf.pth")
            best_loss = test_loss
