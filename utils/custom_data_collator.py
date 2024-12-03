from transformers import PreTrainedTokenizerBase
import torch
from typing import Any, Dict, List

class CustomDataCollatorSameSize:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device='cpu'):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
                
        # Find the maximum length of `input_ids` and `labels` in the batch
        max_length = max(
            max(len(f["input_ids"]) for f in features),
            max(len(f["labels"]) for f in features),
        )

        # Pad `input_ids`, `attention_mask`, and `labels` to the same length
        input_ids = [
            f["input_ids"] + [self.tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
            for f in features
        ]
        attention_mask = [
            [1] * len(f["input_ids"]) + [0] * (max_length - len(f["input_ids"]))
            for f in features
        ]
        labels = [
            f["labels"] + [-100] * (max_length - len(f["labels"]))
            for f in features
        ]

        # print("input_ids: ", input_ids)
        # print("attention_mask: ", attention_mask)
        # print("labels: ", labels)
        
        
        # Create tensors on CPU first
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        
        return batch