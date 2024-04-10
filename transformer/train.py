import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from models.model import Transformer
from utils import (PairDataset,
                   warmup_decay_learningrate, get_data_loader, generate_voc_buffer,
                   model_save, plot_train_test_epoch_loss, plot_train_test_iteration_loss)
from inference import beam_search_decoder, src_example, trg_example
from tqdm import tqdm
from config import TrainingConfig, ModelConfig, LogConfig


def start_train(training_config: TrainingConfig, model_config: ModelConfig, log_config: LogConfig):
    voc_trg = generate_voc_buffer("en")
    voc_src = generate_voc_buffer("de")
    reversed_trg_dict = {v: k for k, v in voc_trg.items()}

    train_src, test_src, train_trg, test_trg = get_data_loader("./data/clean.de", "./data/clean.en", voc_src, voc_trg)
    train_dataset = PairDataset(train_src, train_trg)
    test_dataset = PairDataset(test_src, test_trg)

    train_data_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=training_config.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(src_vocab_num=model_config.src_vocab_num, trg_vocab_num=model_config.trg_vocab_num,
                        max_len=model_config.max_len, embedding_dim=model_config.embedding_dim,
                        stack_num=model_config.stack_num, ffn_dim=model_config.ffn_dim,
                        qkv_dim=model_config.qkv_dim, head_dim=model_config.head_dim, device=device).to(device)

    optimizer = AdamW(model.parameters(), lr=training_config.base_learning_rate)

    iteration_per_epoch = len(train_data_loader)
    warmup_iteration = training_config.warmup_epochs * iteration_per_epoch
    end_iteration = training_config.epochs * iteration_per_epoch
    lr_scheduler = LambdaLR(optimizer,
                            lr_lambda=lambda current_iter: warmup_decay_learningrate(current_iter, warmup_iteration,
                                                                                     end_iteration,
                                                                                     training_config.base_learning_rate,
                                                                                     training_config.max_learning_rate,
                                                                                     training_config.end_learning_rate))

    if training_config.load_checkpoint:
        checkpoint = torch.load(r"D:\pycharm_projects\MachineLearning\transformer\parameters\best_checkpoint.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    trg_pad_idx = voc_trg.get("<pad>", 0)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx, label_smoothing=0.1)

    training_epoch_log_path = 'logs/training_epoch_log.txt'
    train_iteration_log_path = 'logs/train_iteration_log.txt'
    test_iteration_log_path = 'logs/test_iteration_log.txt'

    sentence_demonstration_path = 'logs/sentence_demonstration.txt'
    with open(sentence_demonstration_path, 'a') as sentence_log:
        sentence_log.write(f'original__ sentence: {src_example}\n')
        sentence_log.write(f'translated sentence: {trg_example}\n')

    best_loss = torch.inf
    current_iteration = 0
    train_losses, train_iteration_losses, test_losses, test_iteration_losses = [], [], [], []
    for epoch in range(training_config.epochs):
        model.train()
        train_iteration_loss = 0
        loop = tqdm(enumerate(train_data_loader), desc="Training")
        for index, (source, target) in loop:
            source = source.to(device)  # [batch_size, max_len]
            target = target.to(device)  # [batch_size, max_len]

            output = model(source, target[:, :-1])  # [32, 127, 7000]
            output = output.reshape(-1, output.shape[2])  # [4064, 7000]]
            target = target[:, 1:].reshape(-1)  # [4064]

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            current_iteration += 1
            train_iteration_loss += loss.item()

            train_avg_loss = train_iteration_loss / (index + 1)

            loop.set_postfix(percentage=str(np.round((index + 1) / len(train_data_loader) * 1e2, 2)) + "%",
                             epoch=str(epoch + 1) + "/" + str(training_config.epochs),
                             amount=len(train_data_loader),
                             loss="{:.6e}".format(loss.item()),
                             avg_loss="{:.6e}".format(train_avg_loss),
                             lr="{:.2e}".format(optimizer.param_groups[0]['lr']))
            if log_config.train_iteration_log:
                with open(train_iteration_log_path, 'a') as train_iteration_log:
                    train_iteration_log.write(f'iteration: {index + 1}, loss: {loss.item()}\n')
            train_iteration_losses.append(loss.item())

        train_loss = train_iteration_loss / len(train_data_loader)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            test_iteration_loss = 0
            loop = tqdm(enumerate(test_data_loader), desc="Testing_")
            for index, (source, target) in loop:
                source = source.to(device)
                target = target.to(device)

                output = model(source, target[:, :-1])
                output = output.reshape(-1, output.shape[2])
                target = target[:, 1:].reshape(-1)

                loss = criterion(output, target)
                test_iteration_loss += loss.item()

                test_avg_loss = test_iteration_loss / (index + 1)

                loop.set_postfix(amount=len(test_data_loader),
                                 avg_loss="{:.6e}".format(test_avg_loss),
                                 loss="{:.6e}".format(loss.item()))
                if log_config.test_iteration_log:
                    with open(test_iteration_log_path, 'a') as test_iteration_log:
                        test_iteration_log.write(f'iteration: {index + 1}, loss: {loss.item()}\n')
                test_iteration_losses.append(loss.item())

            test_loss = test_iteration_loss / len(test_data_loader)
            test_losses.append(test_loss)

            if log_config.epoch_log:
                with open(training_epoch_log_path, 'a') as epoch_log:
                    epoch_log.write(f'epoch: {epoch}, train_loss: {train_loss}, test_loss: {test_loss}\n')

        if log_config.plot_fig:
            plot_train_test_epoch_loss(train_losses, test_losses)
            plot_train_test_iteration_loss(train_iteration_losses)
        best_loss = model_save(model, optimizer, lr_scheduler, test_loss, best_loss, epoch, training_config.epochs,
                               log_config.save_epoch_model)

        if log_config.sentence_demonstration:
            predicting_sentence = beam_search_decoder(model, src_example, device=device).cpu().numpy()
            predicting_sentence = [reversed_trg_dict.get(predicting_sentence[i], "<unk>") for i in
                                   range(len(predicting_sentence))][1:-1]
            with open(sentence_demonstration_path, 'a') as sentence_log:
                sentence_log.write(f'epoch: {epoch:}   predicting: {predicting_sentence}\n')
