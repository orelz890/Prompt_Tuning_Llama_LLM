import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from datasets import load_dataset, DatasetDict


from utils.DatasetProcessor import DatasetProcessor

class Aviman1DatasetProcessor(DatasetProcessor):

    def tokenize_function(self, examples):
        """
        Tokenization logic for a specific dataset.
        """
        
        # Convert questions and answers to strings
        questions = [str(q) if q is not None else "" for q in examples["Questions"]]
        answers = [str(a) if a is not None else "" for a in examples["Answers"]]

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
