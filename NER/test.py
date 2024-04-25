import torch
import transformers
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from NER.utils import process_data, EntityDataset
from model import NER_Recognizer

if __name__ == "__main__":
    sentences, tag, id_to_tag_dict = process_data("data/data.csv")
    num_tag = len(id_to_tag_dict.keys())

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_sentences, teva_sentences, train_tag, teva_tag = train_test_split(sentences, tag, test_size=0.1)
    test_sentences, valid_sentences, test_tag, valid_tag = train_test_split(teva_sentences, teva_tag, test_size=0.5)

    test_dataset = EntityDataset(texts=test_sentences, tags=test_tag, tokenizer=tokenizer)
    test_data_loader = DataLoader(test_dataset, batch_size=96)

    device = torch.device("cuda")
    model = NER_Recognizer(num_tag=num_tag)
    model.load_state_dict(torch.load('parameter/nerc_best_checkpoint.pth'))
    model.to(device)

    model.eval()

    y_labels = []
    y_preds = []

    model.eval()

    for data in tqdm(test_data_loader, total=len(test_data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)

        for i in range(len(data["mask"])):
            mask = data["mask"][i].cpu().numpy()
            target = data["target_tag"][i].cpu().numpy()
            temp = []
            for j in range(len(mask)):
                if mask[j] == 1:
                    temp.append(target[j])
            y_labels.append(temp)

        y_pred = model(**data)

        for sentence in y_pred:
            y_preds.append(sentence)

    y_true, y_pred = [], []
    for tags in y_labels:
        for tag in tags:
            y_true.append(tag)

    for tags in y_preds:
        for tag in tags:
            y_pred.append(tag)

    target_names = id_to_tag_dict.values()

    print(classification_report(y_true, y_pred, target_names=target_names))
