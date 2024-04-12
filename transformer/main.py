from train import start_train
from config import TrainingConfig, ModelConfig, LogConfig

if __name__ == "__main__":
    train_config = TrainingConfig(batch_size=128,
                                  epochs=25,
                                  warmup_iteration=2000,
                                  base_learning_rate=1e-10,
                                  max_learning_rate=5e-4,
                                  end_learning_rate=1e-7,
                                  lr_decay_strategy="noam_decay",
                                  label_smoothing=0.1,
                                  load_checkpoint=False,
                                  model_path=r"D:\pycharm_projects\MachineLearning\transformer\parameters\iteration_300_checkpoint.pth",
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

    start_train(train_config, model_config, log_config)
