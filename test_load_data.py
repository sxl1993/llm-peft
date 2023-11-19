#!/usr/bin/env python3
# coding=utf-8

from datasets import load_dataset
from transformers import AutoTokenizer


train_file = "AdvertiseGen/train.json"
validation_file = "AdvertiseGen/dev.json"
test_file = None
data_files = {}
if train_file is not None:
    data_files["train"] = train_file
    extension = train_file.split(".")[-1]
if validation_file is not None:
    data_files["validation"] = validation_file
    extension = validation_file.split(".")[-1]
if test_file is not None:
    data_files["test"] = test_file
    extension = test_file.split(".")[-1]

print(data_files)

cache_dir = None
use_auth_token = False
raw_datasets = load_dataset(
    extension,
    data_files=data_files,
    cache_dir=cache_dir,
    use_auth_token=True if use_auth_token else None
)

# print(raw_datasets)

train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["validation"]

preprocessing_num_workers = 20
column_names = raw_datasets["train"].column_names
overwrite_cache = True

model_name_or_path = "THUDM/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
max_source_length = 64
max_target_length = 128
prompt_column  = "content" 
response_column = "summary"
history_column = None
source_prefix = ""
prefix = source_prefix if source_prefix is not None else ""
ignore_pad_token_for_loss = True

def preprocess_function_train(examples):
    # print("examples: ", examples)
    max_seq_length = max_source_length + max_target_length + 1

    model_inputs = {
        "input_ids": [],
        "labels": [],
    }
    print(len(examples[prompt_column]))
    for i in range(len(examples[prompt_column])):
        if examples[prompt_column][i] and examples[response_column][i]:
            query, answer = examples[prompt_column][i], examples[response_column][i]

            history = examples[history_column][i] if history_column is not None else None
            # prompt = tokenizer.build_prompt(query, history)

            # prompt = prefix + prompt
            prompt = query
            # print("prompt:", prompt)
            # print("response:", answer)
            a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                        max_length=max_source_length)
            b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                        max_length=max_target_length)

            context_length = len(a_ids)
            input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
            labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]
            
            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)

    return model_inputs

def preprocess_function_eval(examples):
    inputs, targets = [], []
    for i in range(len(examples[prompt_column])):
        if examples[prompt_column][i] and examples[response_column][i]:
            query = examples[prompt_column][i]
            history = examples[history_column][i] if history_column is not None else None
            # prompt = tokenizer.build_prompt(query, history)
            prompt = query
            inputs.append(prompt)
            targets.append(examples[response_column][i])

    
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
    
    if ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def print_dataset_example(example):
    print("input_ids \n", example["input_ids"], len(example["input_ids"]))
    print("inputs \n", tokenizer.decode(example["input_ids"]))
    print("label_ids \n", example["labels"], len(example["labels"]))
    print("labels \n", tokenizer.decode(example["labels"]))

# print(type(train_dataset), len(train_dataset), train_dataset)

# train_dataset = eval_dataset.map(
#     preprocess_function_train,
#     batched=True,
#     num_proc=preprocessing_num_workers,
#     remove_columns=column_names,
#     load_from_cache_file=not overwrite_cache,
#     desc="Running tokenizer on train dataset",
# )

# print(type(train_dataset), len(train_dataset))

from data import InputOutputTrainDataset

train_dataset = InputOutputTrainDataset(train_dataset, 
                                        tokenizer,
                                        preprocessing_num_workers=10
                                        max_source_length=64, 
                                        max_target_length=128, 
                                        )

print(train_dataset[0])