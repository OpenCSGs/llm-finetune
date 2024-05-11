from typing import Dict

import numpy as np
import torch
from ._base import BaseFT
from llmadmin.backend.server.models import FTApp

from datasets import load_dataset
from transformers import AutoTokenizer
import ray.data
import torch
import numpy as np

from datasets import load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

import ray.train
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig, FailureConfig
from llmadmin.backend.logger import get_logger

logger = get_logger(__name__)

# GLUE_TASKS = [
#     "cola",
#     "mnli",
#     "mnli-mm",
#     "mrpc",
#     "qnli",
#     "qqp",
#     "rte",
#     "sst2",
#     "stsb",
#     "wnli",
# ]

class RayTrain(BaseFT):
    
    def __init__(self, ftApp: FTApp):
        self.init_model_dataset()
        super().__init__(ftapp=ftApp)
    
    def init_model_dataset(self):
        self.use_gpu = False  # set this to False to run on CPUs
        self.num_workers = 2  # set this to number of GPUs or CPUs you want to use
        logger.info(f"Is CUDA available: {torch.cuda.is_available()}")
        logger.info(f"init model and dataset with num_workers={self.num_workers}, use_gpu={self.use_gpu}")
        self.task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mnli-mm": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),
        }
        self.task = "cola"
        self.actual_task = "mnli" if self.task == "mnli-mm" else self.task
        self.model_checkpoint = "/Users/hhwang/models/distilbert-base-uncased"
        
        logger.info(f"begin load model {self.model_checkpoint}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        self.num_labels = 3 if self.task.startswith("mnli") else 1 if self.task == "stsb" else 2
        self.batch_size = 2
        
        dataset_path = "glue"
        logger.info(f"begin load dataset {dataset_path} -> {self.actual_task}")
        datasets = load_dataset(dataset_path, self.actual_task)
        logger.info(f"loaded datasets: {datasets}")
        item_count = 20
        logger.info(f"convert {item_count} records to ray dataset")
        self.ray_datasets = {
            "train": ray.data.from_huggingface(datasets["train"].select(range(item_count))),
            "validation": ray.data.from_huggingface(datasets["validation"].select(range(item_count))),
            "test": ray.data.from_huggingface(datasets["test"].select(range(item_count))),
        }
        self.train_count = self.ray_datasets["train"].count()
        self.validation_count = self.ray_datasets["validation"].count()
        self.test_count = self.ray_datasets["test"].count()
        logger.info(f"dataset      train count: {self.train_count}")
        logger.info(f"dataset validation count: {self.validation_count}")
        logger.info(f"dataset       test count: {self.test_count}")
        model_name = self.model_checkpoint.split("/")[-1]
        self.name = f"{model_name}-finetuned-{self.task}"
        logger.info(f"output model dir: {self.name}")
    
    # Tokenize input sentences
    def collate_fn(self, examples: Dict[str, np.array]):
        sentence1_key, sentence2_key = self.task_to_keys[self.task]
        if sentence2_key is None:
            outputs = self.tokenizer(
                list(examples[sentence1_key]),
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )
        else:
            outputs = self.tokenizer(
                list(examples[sentence1_key]),
                list(examples[sentence2_key]),
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )
        outputs["labels"] = torch.LongTensor(examples["label"])
        
        if self.use_gpu:
            # Move all input tensors to GPU
            for key, value in outputs.items():
                outputs[key] = value.cuda()

        return outputs

    def train_func(self, config):
        # Calculate the maximum steps per epoch based on the number of rows in the training dataset.
        # Make sure to scale by the total number of training workers and the per device batch size.
        max_steps_per_epoch = self.ray_datasets["train"].count() // (self.batch_size * self.num_workers)
        logger.info(f"max_steps_per_epoch: {max_steps_per_epoch}, batch_size: {self.batch_size}, num_workers: {self.num_workers}")

        # metric = load_metric("glue", self.actual_task)
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint, num_labels=self.num_labels
        )

        train_ds = ray.train.get_dataset_shard("train")
        eval_ds = ray.train.get_dataset_shard("eval")

        train_ds_iterable = train_ds.iter_torch_batches(
            batch_size=self.batch_size, collate_fn=self.collate_fn
        )
        eval_ds_iterable = eval_ds.iter_torch_batches(
            batch_size=self.batch_size, collate_fn=self.collate_fn
        )

        args = TrainingArguments(
            self.name,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=config.get("learning_rate", 2e-5),
            num_train_epochs=config.get("epochs", 2),
            weight_decay=config.get("weight_decay", 0.01),
            push_to_hub=False,
            max_steps=max_steps_per_epoch * config.get("epochs", 2),
            disable_tqdm=True,  # declutter the output a little
            use_cpu=not self.use_gpu,  # you need to explicitly set no_cuda if you want CPUs
            report_to="none",
        )

        # def compute_metrics(eval_pred):
        #     predictions, labels = eval_pred
        #     if self.task != "stsb":
        #         predictions = np.argmax(predictions, axis=1)
        #     else:
        #         predictions = predictions[:, 0]
        #     return metric.compute(predictions=predictions, references=labels)

        trainer = Trainer(
            model,
            args,
            train_dataset=train_ds_iterable,
            eval_dataset=eval_ds_iterable,
            tokenizer=tokenizer,
            # compute_metrics=compute_metrics,
        )

        trainer.add_callback(RayTrainReportCallback())

        trainer = prepare_trainer(trainer)

        logger.info("Starting training")
        trainer.train()

    def train(self):
        # metric_name = (
        #     "pearson"
        #     if self.task == "stsb"
        #     else "matthews_correlation"
        #     if self.task == "cola"
        #     else "accuracy"
        # )
        
        # validation_key = (
        #     "validation_mismatched"
        #     if self.task == "mnli-mm"
        #     else "validation_matched"
        #     if self.task == "mnli"
        #     else "validation"
        # )
        logger.info(f"build ray TorchTrainer")
        
        trainer = TorchTrainer(
            self.train_func,
            scaling_config=ScalingConfig(num_workers=self.num_workers, use_gpu=self.use_gpu),
            datasets={
                "train": self.ray_datasets["train"],
                "eval": self.ray_datasets["validation"],
            },
            run_config=RunConfig(
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute="eval_loss",
                    checkpoint_score_order="min",
                ),
                failure_config=FailureConfig(
                    max_failures=5
                )
            ),
        )
        
        logger.info(f"begin ray train fit")
        result = trainer.fit()
        logger.info(f"end ray train fit")
        logger.info(f"result: {result}")

