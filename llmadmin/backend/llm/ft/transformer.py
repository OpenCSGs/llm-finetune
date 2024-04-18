from ._base import BaseFT
from abc import ABC, abstractmethod
from llmadmin.backend.logger import get_logger
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from typing import Union
from llmadmin.backend.server.models import FTApp
from datasets import load_dataset
from datasets import load_metric
import pandas as pd
from ray.data.preprocessors import BatchMapper
import ray
import torch
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer
import numpy as np
from ray.train.huggingface import TransformersTrainer
from ray.air.config import RunConfig, CheckpointConfig
from .utils import parse_task_name 
from .tasks import TASK_REGISTRY
from .tasks._base import Task
from .methods.base import get_train_model
from ray.train.huggingface import TransformersCheckpoint
from .const import CHECKPOINT_PATH
from .callback import CustomCallback

from llmadmin.backend.llm.utils import ( initialize_node )

logger = get_logger(__name__)

class TransformersFT(BaseFT):
    def __init__(self, ftApp: FTApp):
        super().__init__(ftapp=ftApp)
    
    def train(self):
        self.trainV2()
    
    # Transformer train only
    def trainV2(self):
        taskobj: Task = None
        task = parse_task_name(self.ftapp)
        logger.info(f"TransformersFT.trainV2 finetune task name: '{task}'")
        taskcls = TASK_REGISTRY[task]

        if not taskcls:
            logger.error(f"Couldn't load defined task from register: '{task}'")
            raise
        
        logger.info("Start initializing finetune node tasks")
        initialize_node(self.model_config.model_id, self.model_config.initialization.s3_mirror_config)
        logger.info(f"Start loading tokenizer for finetune {self.model_config.model_id}")
        # self.model_config.model_id = '/root/.cache/huggingface/hub/ZhipuAI/chatglm3-6b/'
        # self.model_config.model_id = '/data/hhwang/models/chatglm2-6b/'
        tokenizer = self.initializer.load_tokenizer(self.model_config.model_id)
        if self.model_config.add_special_tokens:
            add_special_tokens = self.model_config.add_special_tokens
            if add_special_tokens.get("pad_token"):
                tokenizer.pad_token = add_special_tokens.get("pad_token")
            if add_special_tokens.get("eos_token"):
                tokenizer.eos_token = add_special_tokens.get("eos_token")
        logger.info(f"Initialize {taskcls} and load dataset")
        # logger.info(f"Initialize {taskcls} and load dataset for model {self.model_config.model_id}")
        taskobj = taskcls.from_tokenizer(tokenizer, self.ftapp.ft_config)
        logger.info(f"Load model {self.model_config.model_id} by {taskobj.AUTO_MODEL_CLASS}")
        from_pretrained_kwargs = taskobj.FROM_PRETRAINED_KWARGS if taskobj.FROM_PRETRAINED_KWARGS else {}
        model = self.initializer.load_model(self.model_config.model_id, taskobj.AUTO_MODEL_CLASS, **from_pretrained_kwargs)
        if self.model_config.quantization_bit is not None:
            print(f"Quantized to {self.model_config.quantization_bit} bit")
            model = model.quantize(self.model_config.quantization_bit)
        
        taskobj.set_model(model)
        
        # preprocess_function = taskobj.get_data_proprocess()
        # compute_metrics_function = taskobj.get_compute_metrics()
        data_collator = taskobj.get_data_collator()
        # batch_encoder = BatchMapper(preprocess_function, batch_format="pandas")
        
        data_config = self.ftapp.ft_config.data_config
        use_gpu = True if torch.cuda.is_available() else False
        use_mps = True if torch.backends.mps.is_available() else False
        logger.info(f"use_gpu: {use_gpu}, use_cpu: {not use_gpu}, use_mps: {use_mps}")
 
        logger.info(f"Finetune get train and validation dataset")
        if data_config.num_row > 0:
            # only for test purpose
            train_dataset = taskobj.getSmallTrainDataSet(data_config.num_row)
            eval_dataset = taskobj.getSmallEvalDataSet(data_config.num_row)
        else:
            # For train
            train_dataset = taskobj.getTrainDataSet()
            eval_dataset = taskobj.getEvalDataSet()
        
        logger.info(f"Finetune train dataset {train_dataset}")
        logger.info(f"Finetune eval dataset {eval_dataset}")
        
        if hasattr(model, "is_parallelizable"):
            logger.info(f"model.is_parallelizable = {model.is_parallelizable}")
            
        if hasattr(model, "model_parallel"):
            logger.info(f"model.model_parallel = {model.model_parallel}")
        
        if getattr(model, "hf_device_map", None) is not None:
            logger.info(f"model.hf_device_map is {model.hf_device_map}")
        
        ftConfig = self.ftapp.ft_config.train_config.base_config
        model_name = self.model_config.model_id.split("/")[-1]
        task_name = self.ft_task
        outputDir = f"{ftConfig.checkpoints_output_dir}/{model_name}-finetuned-{task_name}-{data_config.data_path}-{data_config.subset}"
        logger.info(f"Finetune checkpoints output dir: {outputDir}")
        args = TrainingArguments(
            outputDir,
            evaluation_strategy=ftConfig.evaluation_strategy,
            save_strategy=ftConfig.save_strategy,
            logging_strategy=ftConfig.logging_strategy,
            logging_steps = 2,
            save_steps = ftConfig.save_steps,
            eval_steps = 2,
            learning_rate=ftConfig.learning_rate,
            per_device_train_batch_size=ftConfig.per_device_train_batch_size,
            per_device_eval_batch_size=ftConfig.per_device_eval_batch_size,
            num_train_epochs=ftConfig.num_train_epochs,
            weight_decay=ftConfig.weight_decay,
            push_to_hub=False,
            disable_tqdm=False,  # declutter the output a little
            use_cpu=not use_gpu,  # you need to explicitly set no_cuda if you want CPUs
            remove_unused_columns=ftConfig.remove_unused_columns,
        )
        trainConfig = self.ftapp.ft_config.train_config
        ftMethod = self.ftapp.ft_config.ft_method
        model = get_train_model(model, ftMethod, trainConfig)
        trainer = Trainer(
        # trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            # compute_metrics=compute_metrics_function,
            data_collator=data_collator,
        )
        trainer.add_callback(CustomCallback(trainer))
        logger.info("Starting training")
        trainResult = trainer.train()
        logger.info(f"Train result {trainResult}")
        trainer.save_model()
        logger.info(f"Save model to {trainer.args.output_dir}")
        logger.info("Done training")
        
    # depend on ray for distribution    
    def trainV1(self):
        taskobj: Task = None
        task = parse_task_name(self.ftapp)
        logger.info(f"TransformersFT.trainV1 finetune task name {task}")
        taskcls = TASK_REGISTRY[task]

        if not taskcls:
            logger.error(f"Couldn't load defined task from register: {task}")
            raise
        
        logger.info("Starting initialize Finetune node tasks")
        initialize_node(self.model_config.model_id, self.model_config.initialization.s3_mirror_config)
        
        tokenizer = self.initializer.load_tokenizer(self.model_config.model_id)
        logger.info("Done load tokenizer for finetune")
        
        taskobj = taskcls.from_tokenizer(tokenizer, self.ftapp.ft_config)
        
        from_pretrained_kwargs = taskobj.FROM_PRETRAINED_KWARGS if taskobj.FROM_PRETRAINED_KWARGS else {}
        model = self.initializer.load_model(self.model_config.model_id, taskobj.AUTO_MODEL_CLASS, **from_pretrained_kwargs)
        taskobj.set_model(model)

        preprocess_function = taskobj.get_data_proprocess()
        compute_metrics_function = taskobj.get_compute_metrics()
        data_collator = taskobj.get_data_collator()
        batch_encoder = BatchMapper(preprocess_function, batch_format="pandas")
        
        ray_datasets = ray.data.from_huggingface(taskobj.get_dataset())
        model_name = self.model_config.model_id.split("/")[-1]
        task = self.ft_task
        name = f"{model_name}-finetuned-{task}"
        use_gpu = True if torch.cuda.is_available() else False

        def trainer_init_per_worker(train_dataset, eval_dataset = None, **config):
            print(f"Is CUDA available: {torch.cuda.is_available()}")

            args = TrainingArguments(
                name,
                evaluation_strategy=config.get("evaluation_strategy", "epoch"),
                save_strategy=config.get("save_strategy", "epoch"),
                logging_strategy=config.get("logging_strategy", "epoch"),
                logging_steps = 2,
                save_steps = 500,
                eval_steps = 2,
                learning_rate=config.get("learning_rate", 2e-5),
                per_device_train_batch_size=config.get("per_device_train_batch_size", 16),
                per_device_eval_batch_size=config.get("per_device_train_batch_size", 16),
                num_train_epochs=config.get("epochs", 2),
                weight_decay=config.get("weight_decay", 0.01),
                push_to_hub=False,
                disable_tqdm=False,  # declutter the output a little
                no_cuda=not use_gpu,  # you need to explicitly set no_cuda if you want CPUs
                remove_unused_columns=config.get("remove_unused_columns", True),
                fp16=True,
            )

            trainer = Trainer(
                model,
                args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics_function,
                data_collator=data_collator,
            )
            trainer.add_callback(CustomCallback(trainer))
            print("Starting training")

            return trainer
        
        trainer = TransformersTrainer(
            trainer_init_per_worker=trainer_init_per_worker,
            trainer_init_config = self.train_conf.get_train_kwargs(),
            scaling_config=self.scale_config.as_air_scaling_config(),
            datasets={
                "train": ray_datasets[taskobj.training_key()],
                "evaluation": ray_datasets[taskobj.validation_key()],
            },
            run_config=RunConfig(
                # callbacks=[MLflowLoggerCallback(experiment_name=name)],
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute="eval_loss",
                    checkpoint_score_order="min",
                ),
            ),
            preprocessor=batch_encoder,
        )

        result = trainer.fit()
        print(result)
        checkpoint = TransformersCheckpoint.from_checkpoint(result.checkpoint)
        hf_trainer = checkpoint.get_model(model=taskobj.AUTO_MODEL_CLASS)
        hf_trainer.save_pretrained(CHECKPOINT_PATH)
        tokenizer.save_pretrained(CHECKPOINT_PATH)

        print("Done")




