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
            examples: A batch of dataset examples containing "Questions" and "Answers" fields.

        Returns:
            dict: Tokenized inputs and labels.
        """
            
        # Convert questions and answers to strings
        questions = ["[User]: " + str(q) + "\n[MODEL]: " for q in examples["Questions"]]
        answers = [str(a) for a in examples["Answers"]]
        

        # Tokenize inputs (questions)
        inputs = self.tokenizer(
            questions,
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )

        # Tokenize targets (answers)
        labels = self.tokenizer(
            text_target=answers,
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )["input_ids"]

        # Add labels to inputs
        inputs["labels"] = labels
        return inputs

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

        return self.clean_dataset(dataset=dataset)
