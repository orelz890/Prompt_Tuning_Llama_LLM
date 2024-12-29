import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from datasets import load_dataset, DatasetDict


from utils.DatasetProcessor import DatasetProcessor

class Aviman1DatasetProcessor(DatasetProcessor):
    """
    A dataset processor for the Aviman1 dataset.
    Overrides the DatasetProcessor base class to handle dataset-specific tokenization logic.

    Methods:
        tokenize_function: Overrides to tokenize input questions and target answers.
    """
    
    def tokenize_function(self, examples):
        """
        Overrides DatasetProcessor.tokenize_function.
        Tokenization logic for the Aviman1 dataset.

        Args:
            examples: A batch of dataset examples containing "Questions" and "Answers", "prompt" fields.

        Returns:
            Tensor: Tokenized inputs and labels.
        """

        tokens = self.tokenizer(examples['prompt'], padding="max_length", truncation=True, max_length=128)
        # Set padding token labels to -100 to ignore them in loss calculation
        tokens['labels'] = [
            -100 if token == self.tokenizer.pad_token_id else token for token in tokens['input_ids']
        ]
        
        return tokens

    def load_dataset(self):
        """
        Load the dataset using Hugging Face's `load_dataset`.

        Returns:
            DatasetDict: Loaded dataset.
        """
        
        print(f"[INFO] Loading dataset from {self.dataset_path}")
        dataset = load_dataset(self.dataset_path)

        # Drop the 'Unnamed: 2' column
        dataset = dataset.remove_columns(['Unnamed: 2'])
        dataset = self.clean_dataset(dataset=dataset)
        
        # Create the instruct prompt
        new_dataset = dataset.map(self.apply_chat_template)
        
        return new_dataset
