import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from datasets import load_dataset, DatasetDict, Dataset


from utils.DatasetProcessor import DatasetProcessor

class GooglePersonaDatasetProcessor(DatasetProcessor):
    """
    A dataset processor for the Google Persona Chat dataset.
    Overrides the methods from the DatasetProcessor base class to handle dataset-specific
    tokenization and preprocessing logic.

    Methods:
        train_eval_test_split: Overrides to split the dataset into train, validation, and test sets.
        tokenize_function: Overrides to tokenize input and target sentences.
        get_datasets: Overrides to process raw datasets and extract user conversations.
    """
    
    def train_eval_test_split(self):
        """
        Overrides DatasetProcessor.train_eval_test_split.
        Splits the dataset into train, validation, and test sets.

        Returns:
            tuple: Train, validation, and test datasets.
        """
        
        dataset = self.load_dataset()
        print(len(dataset["train"]))
        return dataset["train"], dataset["validation"], dataset["test"]
        
    def tokenize_function(self, examples):
        """
        Overrides DatasetProcessor.tokenize_function.
        Tokenization logic for the Google Persona Chat dataset.

        Args:
            examples: A batch of dataset examples containing input and target sentences.

        Returns:
            dict: Tokenized inputs and labels.
        """

        user1_sentences = examples["input_sentences"]
        user2_sentences = examples["target_sentences"]

        # Tokenize inputs (questions)
        inputs = self.tokenizer(
            user1_sentences,
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )

        # Tokenize targets (answers)
        labels = self.tokenizer(
            text_target=user2_sentences,
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )["input_ids"]

        # Add labels to inputs
        inputs["labels"] = labels
    
        return inputs

    
    def get_datasets(self, raw_dataset):
        """
        Overrides DatasetProcessor.get_datasets.
        Processes raw datasets to extract user conversations and structure them
        into input-output sentence pairs.

        Args:
            raw_dataset: The raw dataset containing user conversations.

        Returns:
            Dataset: A structured dataset with input and target sentences.
        """
        
        user1_sentences = []
        user2_sentences = []
        
        for index, conversation in enumerate(raw_dataset["Best Generated Conversation"]):
            if index > 150:
                break
            
            if conversation != None:
                sentences = conversation.split('\n')
                
                # Remove last sentence
                if len(sentences) % 2 != 0:
                    sentences = sentences[:-1]
                
                user1_sentences.extend([s.removeprefix("User 1: ") for index, s in enumerate(sentences) if index % 2 == 0])
                user2_sentences.extend([s.removeprefix("User 2: ") for index, s in enumerate(sentences) if index % 2 != 0])
        
        # Create a Dataset object
        return Dataset.from_dict({
            "input_sentences": user1_sentences,
            "target_sentences": user2_sentences
        })
            
    