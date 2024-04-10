from train import start_train
from config import TrainingConfig, ModelConfig, LogConfig

if __name__ == "__main__":
    train_config = TrainingConfig(batch_size=128,
                                  epochs=20,
                                  warmup_epochs=1,
                                  base_learning_rate=1e-10,
                                  max_learning_rate=3e-4,
                                  end_learning_rate=1e-10,
                                  label_smoothing=0.1,
                                  load_checkpoint=True)
    model_config = ModelConfig(src_vocab_num=7000,
                               trg_vocab_num=7000,
                               max_len=128,
                               embedding_dim=128,
                               stack_num=3,
                               qkv_dim=32,
                               ffn_dim=512,
                               head_dim=4)
    log_config = LogConfig(epoch_log=True,
                           train_iteration_log=True,
                           test_iteration_log=False,
                           plot_fig=True,
                           save_epoch_model=1,
                           sentence_demonstration=True)

    start_train(train_config, model_config, log_config)
