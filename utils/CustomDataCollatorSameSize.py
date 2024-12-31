from transformers import PreTrainedTokenizerBase
import torch
from typing import Any, Dict, List

class CustomDataCollatorSameSize:
    """
        A custom data collator for padding sequences in a batch to the same length.
        Ensures all `input_ids`, `attention_mask`, and `labels` in the batch are of equal length.

        Attributes:
            tokenizer (PreTrainedTokenizerBase): The tokenizer used for padding and managing special tokens.
            device (str): The device (e.g., 'cpu', 'cuda') where tensors will be moved. Default is 'cpu'.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device='cpu'):
        """
            Initialize the data collator with a tokenizer and device.

            Args:
                tokenizer (PreTrainedTokenizerBase): The tokenizer used for managing special tokens and padding.
                device (str): The device for tensor operations ('cpu' or 'cuda'). Default is 'cpu'.
        """
        
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
            Collate a batch of features by padding `input_ids`, `attention_mask`, and `labels` to the same length.

            Args:
                features (List[Dict[str, Any]]): A list of feature dictionaries containing `input_ids` and `labels`.

            Returns:
                Dict[str, torch.Tensor]: A dictionary containing padded tensors for `input_ids`, `attention_mask`, and `labels`.
        """
                      
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
            f["labels"] + [self.tokenizer.pad_token_id] * (max_length - len(f["labels"]))
            for f in features
        ]
        
        
        # Create tensors on CPU first
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        
        return batch
