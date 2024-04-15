from peft import get_peft_model
from llmadmin.backend.logger import get_logger

logger = get_logger(__name__)

def get_trainable_parameters(model):
    """
    get the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def lora_model(model, lora_config):
    logger.info("Load lora config")
    logger.info(lora_config)
    # from peft import LoraConfig, TaskType
    # lora_config = LoraConfig(
    #     task_type=TaskType.SEQ_CLS, r=1, lora_alpha=1, lora_dropout=0.1
    # )
    # logger.info(lora_config)
    lora_config.loftq_config = {}
    logger.info("Using peft to avoid Catastrophic Forgetting")
    model = get_peft_model(model, lora_config)
    get_trainable_parameters(model)
    return model
