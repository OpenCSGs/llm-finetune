from .lora import lora_model
from llmadmin.backend.logger import get_logger

logger = get_logger(__name__)

def get_train_model(model, ft_method, trainConfig):
    if ft_method == "lora":
        lora_config = trainConfig.lora_config
        model = lora_model(model, lora_config)
    return model
