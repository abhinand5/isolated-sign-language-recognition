import tensorflow as tf
from src.islr.logging import get_logger

logger = get_logger(__name__)


# Custom callback to update weight decay with learning rate
class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, wd_ratio):
        self.step_counter = 0
        self.wd_ratio = wd_ratio

    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.weight_decay = (
            self.model.optimizer.learning_rate * self.wd_ratio
        )
        logger.info(
            f"Starting Epoch {epoch} with learning rate: {self.model.optimizer.learning_rate.numpy():.2e}, weight decay: {self.model.optimizer.weight_decay.numpy():.2e}"
        )
