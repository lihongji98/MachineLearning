from train import start_train
from config import TrainingConfig, ModelConfig, LogConfig

if __name__ == "__main__":
    train_config = TrainingConfig(batch_size=128,
                                  epochs=100,
                                  warmup_epochs=2,
                                  base_learning_rate=1e-10,
                                  max_learning_rate=1e-3,
                                  end_learning_rate=1e-7,
                                  label_smoothing=0.1,
                                  load_checkpoint=False,
                                  model_path=r"D:\pycharm_projects\MachineLearning\transformer\parameters\best_checkpoint.pth")
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
                           test_iteration_log=False,
                           plot_fig=True,
                           save_epoch_model=1,
                           save_iteration_model=15000,
                           sentence_demonstration=True)

    start_train(train_config, model_config, log_config)
