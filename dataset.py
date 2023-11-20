from datasets import arrow_dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset

def print_dataset_example(example, tokenizer):
    print("input_ids \n", example["input_ids"])
    print("inputs \n", tokenizer.decode(example["input_ids"]))
    print("label_ids \n", example["labels"])
    print("labels \n", tokenizer.decode(example["labels"]))


class InputOutputTrainDataset(Dataset):
    def __init__(self,
                 data: arrow_dataset.Dataset,
                 tokenizer: PreTrainedTokenizer,
                 preprocessing_num_workers: int,
                 max_source_length: int,
                 max_target_length: int,
                 prompt_column: str = "content",
                 response_column: str = "summary",
                 history_column: str = None,
                 ignore_pad_token_for_loss: bool=True,
                 overwrite_cache: bool=True
                 ):
        super(InputOutputTrainDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.preprocessing_num_workers = preprocessing_num_workers
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_source_length + max_target_length + 1
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        
        self.data = self.data.map(self._preprocess_data,
                                  batched=True,
                                  num_proc=self.preprocessing_num_workers,
                                  remove_columns=data.column_names,
                                  load_from_cache_file=not overwrite_cache,
                                  desc="Running tokenizer on train dataset")
    
    def _preprocess_data(self, data_item):
        model_inputs = {
        "input_ids": [],
        "labels": [],
        }
        for i in range(len(data_item[self.prompt_column])):
            if data_item[self.prompt_column][i] and data_item[self.response_column][i]:
                prompt = data_item[self.prompt_column][i]
                response = data_item[self.response_column][i]
                history = data_item[self.history_column][i] if self.history_column is not None else None
                
                a_ids = self.tokenizer.encode(text=prompt, 
                                              add_special_tokens=True, 
                                              truncation=True, 
                                              max_length=self.max_source_length)
                b_ids = self.tokenizer.encode(text=response,
                                              add_special_tokens=False, 
                                              truncation=True,
                                              max_length=self.max_target_length)
                context_length = len(a_ids)
                input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
                labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]
        
                pad_len = self.max_seq_length - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                labels = labels + [self.tokenizer.pad_token_id] * pad_len
                labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]
                if self.ignore_pad_token_for_loss:
                    labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]
        
                assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)
        return model_inputs
        
    def __getitem__(self, i) -> dict:
        return self.data[i]
       
    def __len__(self) -> int:
        return len(self.data)
    
class InputOutputEvalDataset(Dataset):
    def __init__(self,
                 data: arrow_dataset.Dataset,
                 tokenizer: PreTrainedTokenizer,
                 preprocessing_num_workers: int,
                 max_source_length: int, 
                 max_target_length:int,
                 prompt_column: str = "content",
                 response_column: str = "summary",
                 history_column: str = None,
                 ignore_pad_token_for_loss: bool=True,
                 overwrite_cache: bool=True
                 ):
        super(InputOutputEvalDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.preprocessing_num_workers = preprocessing_num_workers
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_source_length + max_target_length + 1
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.data = self.data.map(self._preprocess_data,
                                  batched=True,
                                  num_proc=self.preprocessing_num_workers,
                                  remove_columns=data.column_names,
                                  load_from_cache_file=not overwrite_cache,
                                  desc="Running tokenizer on validation dataset")
    
    def _preprocess_data(self, data_item):
        inputs, targets = [], []
        for i in range(len(data_item[self.prompt_column])):
            if data_item[self.prompt_column][i] and data_item[self.response_column][i]:
                query = data_item[self.prompt_column][i]
                history = data_item[self.history_column][i] if self.history_column is not None else None
                prompt = query
                inputs.append(prompt)
                targets.append(data_item[self.response_column][i])
    
        model_inputs = self.tokenizer(inputs, 
                                      max_length=self.max_source_length, 
                                      truncation=True, 
                                      padding=True)
        labels = self.tokenizer(text_target=targets, 
                                max_length=self.max_target_length, 
                                truncation=True)
        if self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def __getitem__(self, i) -> dict:
        return self.data[i]
        
    def __len__(self) -> int:
        return len(self.data)
    
if __name__ == "__main__":
    pass
