import os
import threading
import time
import datetime

from train import start_train
from config import TrainingConfig, ModelConfig, LogConfig

train_config = TrainingConfig(batch_size=128,
                              epochs=25,
                              warmup_iteration=4000,
                              base_learning_rate=1e-10,
                              max_learning_rate=5e-4,
                              end_learning_rate=1e-7,
                              lr_decay_strategy="noam_decay",
                              label_smoothing=0.1,
                              load_checkpoint=False,
                              model_path="",
                              train_size=0.9)
model_config = ModelConfig(src_vocab_num=16000,
                           trg_vocab_num=16000,
                           max_len=128,
                           embedding_dim=512,
                           stack_num=6,
                           qkv_dim=64,
                           ffn_dim=2048,
                           head_dim=8)
log_config = LogConfig(epoch_log=True,
                       train_iteration_log=True,
                       plot_fig=False,
                       save_epoch_model=1,
                       save_iteration_model=5022,
                       sentence_demonstration=True)


def main():
    start_train(train_config, model_config, log_config)


def shutdown_timer(timer_duration):
    time.sleep(timer_duration)
    print("Timer expired. Shutting down.")
    os._exit(0)


if __name__ == "__main__":
    current_time = time.time()
    current_datetime = datetime.datetime.fromtimestamp(current_time)
    shutdown_time = datetime.datetime(2024, 4, 13, 3, 47, 0)

    if current_datetime < shutdown_time:
        run_time = (shutdown_time - current_datetime).total_seconds()
        run_time_hour = int(run_time // 3600)
        run_time_minutes = int((run_time % 3600) // 60)
        run_time_seconds = int(run_time % 60)

        print(
            f"The program will shut down in {run_time_hour} hours, {run_time_minutes} minutes, and {run_time_seconds} seconds.")

        main_thread = threading.Thread(target=main)
        main_thread.start()

        timer_thread = threading.Thread(target=shutdown_timer, args=(run_time,))
        timer_thread.start()

        main_thread.join()
    else:
        print("The shutdown time has already passed.")
