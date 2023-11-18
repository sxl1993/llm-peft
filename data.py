from datasets import arrow_dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from typing import List


def print_dataset_example(example, tokenizer):
    print("input_ids \n", example["input_ids"])
    print("inputs \n", tokenizer.decode(example["input_ids"]))
    print("label_ids \n", example["labels"])
    print("labels \n", tokenizer.decode(example["labels"]))
        
class InputOutputTrainDataset(Dataset):
    def __init__(self, data: arrow_dataset.Dataset, tokenizer: PreTrainedTokenizer, max_source_length: int, max_target_length: int):
        super(InputOutputTrainDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_source_length + max_target_length + 1
        self.data = data

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, i) -> dict:
        data_item = self.data[i]
       
        a_ids = self.tokenizer.encode(text=data_item['content'], add_special_tokens=True, truncation=True,
                                         max_length=self.max_source_length)
        b_ids = self.tokenizer.encode(text=data_item['summary'], add_special_tokens=False, truncation=True,
                                    max_length=self.max_target_length)

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]
        
        pad_len = self.max_seq_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": input_ids,
            "labels": labels
        }
        
class InputOutputEvalDataset(Dataset):
    def __init__(self, data: List[dict], tokenizer: PreTrainedTokenizer, max_source_length: int, max_target_length: int, ignore_pad_token_for_loss: bool=True):
        super(InputOutputEvalDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_source_length + max_target_length + 1
        self.data = data
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, i) -> dict:
        data_item = self.data[i]
        
        model_inputs = self.tokenizer(text_target=data_item['content'],
                                      max_length=self.max_source_length, 
                                      truncation=True, 
                                      padding=True
                                      )
        labels = self.tokenizer(text_target=data_item['summary'], 
                                max_length=self.max_target_length, 
                                truncation=True
                                )
        if self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
        
if __name__ == "__main__":
    pass
