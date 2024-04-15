from ._base import Task
from transformers import AutoModel, DataCollatorForSeq2Seq, AutoModelForCausalLM
from typing import Any
import pandas as pd
import numpy as np
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class NoheaderAdvertiseGen(Task):
    # AUTO_MODEL_CLASS = AutoModel
    AUTO_MODEL_CLASS = AutoModelForCausalLM

    DATASET_PATH = "AdvertiseGen"
    prompt_column = "content"
    response_column = "summary"
    # history_column = "history"

    def get_data_proprocess(self) -> Any:
        self.prompt_column = self.ft_config.data_config.input_columns[0]
        self.response_column = self.ft_config.data_config.validation_column
        self.DATASET_PATH = self.ft_config.data_config.data_path
        tokenizer = self.tokenizer
        max_length = self.ft_config.train_config.base_config.max_length
        # adopt python decorator TODO
        def preprocess_function(examples):            
            # examples = examples.to_dict("list")
            #-- start
            max_source_length = int(max_length / 2)
            max_target_length = max_length - max_source_length
            max_source_length = 64
            max_target_length = 128
            max_seq_length = max_source_length + max_target_length + 1

            model_inputs = {
                "input_ids": [],
                "labels": [],
            }
            prefix = ""
            for i in range(len(examples[self.prompt_column])):
                if examples[self.prompt_column][i] and examples[self.response_column][i]:
                    query, answer = examples[self.prompt_column][i], examples[self.response_column][i]

                    # history = examples[history_column][i] if history_column is not None else None
                    # history = None
                    # prompt = tokenizer.build_prompt(query, history)

                    prompt = prefix + query
                    print(f"tokenizer is: {tokenizer}")
                    a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True, padding=True,
                                            max_length=max_source_length)
                    b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True, padding=True,
                                            max_length=max_target_length)

                    context_length = len(a_ids)
                    input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
                    labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]
                    
                    pad_len = max_seq_length - len(input_ids)
                    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                    labels = labels + [tokenizer.pad_token_id] * pad_len
                    
                    # if data_args.ignore_pad_token_for_loss:
                        # labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
                    # labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
                    
                    model_inputs["input_ids"].append(input_ids)
                    model_inputs["labels"].append(labels)

            return model_inputs
        
        return preprocess_function

    def get_eval_preprocess(self) -> Any:
        tokenizer = self.tokenizer
        def preprocess_function_eval(examples):
            max_source_length = 64
            max_target_length = 128
            inputs, targets = [], []
            prefix = ""
            for i in range(len(examples[self.prompt_column])):
                if examples[self.prompt_column][i] and examples[self.response_column][i]:
                    query = examples[self.prompt_column][i]
                    # history = examples[history_column][i] if history_column is not None else None
                    # history = None
                    # prompt = tokenizer.build_prompt(query, history)
                    inputs.append(query)
                    targets.append(examples[self.response_column][i])

            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding=True)
            labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

            # if data_args.ignore_pad_token_for_loss:
            #     labels["input_ids"] = [
            #         [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            #     ]
            model_inputs["labels"] = labels["input_ids"]

            return model_inputs
        
        return preprocess_function_eval

    def get_compute_metrics(self) -> Any:
        tokenizer = self.tokenizer

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            score_dict = {
                "rouge-1": [],
                "rouge-2": [],
                "rouge-l": [],
                "bleu-4": []
            }
            for pred, label in zip(decoded_preds, decoded_labels):
                hypothesis = list(jieba.cut(pred))
                reference = list(jieba.cut(label))
                rouge = Rouge()
                scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
                result = scores[0]
                
                for k, v in result.items():
                    score_dict[k].append(round(v["f"] * 100, 4))
                bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
                score_dict["bleu-4"].append(round(bleu_score * 100, 4))

            for k, v in score_dict.items():
                score_dict[k] = float(np.mean(v))
            return score_dict
        
        return compute_metrics

    def get_data_collator(self) -> Any:
        # label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        label_pad_token_id = self.tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
            # padding=True
            padding=False
        )
        return data_collator
    
    def training_key(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return "train"

    def validation_key(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return "validation"
    
    def getTrainDataSet(self):
        return self.dataset[self.training_key()].map(self.get_data_proprocess(), batched=True, remove_columns=[self.prompt_column, self.response_column])

    def getEvalDataSet(self):
        return self.dataset[self.validation_key()].map(self.get_data_proprocess(), batched=True, remove_columns=[self.prompt_column, self.response_column])

    def getSmallTrainDataSet(self, len: int):
        return self.dataset[self.training_key()].select(range(len)).map(self.get_data_proprocess(), batched=True, remove_columns=[self.prompt_column, self.response_column])

    def getSmallEvalDataSet(self, len: int):
        return self.dataset[self.validation_key()].select(range(len)).map(self.get_data_proprocess(), batched=True, remove_columns=[self.prompt_column, self.response_column])
        # return self.dataset[self.validation_key()].select(range(len)).map(self.get_eval_preprocess(), batched=True, remove_columns=[self.prompt_column, self.response_column])
