from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 256
    train_batch_size = 16
    eval_batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4
