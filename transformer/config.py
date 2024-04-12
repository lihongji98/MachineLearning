from dataclasses import dataclass


@dataclass
class TrainingConfig:
    base_learning_rate: float = 1e-10
    max_learning_rate: float = 1e-3
    end_learning_rate: float = 1e-7
    batch_size: int = 128
    epochs: int = 10
    warmup_iteration: int = 4000
    lr_decay_strategy: str = "noam_decay"
    label_smoothing: float = 0.1
    load_checkpoint: bool = True
    model_path: str = ""
    train_size: float = 0.9


@dataclass
class ModelConfig:
    src_vocab_num: int = 8000
    trg_vocab_num: int = 8000
    max_len: int = 128
    embedding_dim: int = 512
    stack_num: int = 6
    ffn_dim: int = 2048
    qkv_dim: int = 64
    head_dim: int = 8


@dataclass
class LogConfig:
    epoch_log: bool = True
    train_iteration_log: bool = True
    plot_fig: bool = True
    save_epoch_model: int = 3
    save_iteration_model: int = 15000
    sentence_demonstration: bool = True
