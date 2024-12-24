from abc import ABC, abstractmethod
from datasets import load_dataset, DatasetDict


class DatasetProcessor(ABC):
    """
    Abstract base class for dataset processing.
    Requires overriding of `load_dataset`, `preprocess`, and `tokenize_function`.

    Attributes:
        tokenizer: Tokenizer for processing the dataset.
        dataset_path (str): Path to the dataset.
        batch_size (int): Batch size for tokenizing the dataset. Default is 16.
        test_size (float): Fraction of the dataset reserved for testing. Default is 0.2.
        seed (int): Random seed for reproducibility. Default is 42.
        max_length (int): Maximum token length. Default is 512.
    """

    def __init__(self, tokenizer, dataset_path, **kwargs):
        """
        Initialize the DatasetProcessor.

        Args:
            tokenizer: Tokenizer for processing the dataset.
            dataset_path (str): Path to the dataset to be loaded.
            
            **kwargs: Additional parameters for dataset processing, including:
                - batch_size (int): Batch size for tokenizing. Default is 16.
                - test_size (float): Fraction of the dataset reserved for testing. Default is 0.2.
                - seed (int): Random seed for reproducibility. Default is 42.
                - max_length (int): Maximum token length. Default is 512.

        Raises:
            ValueError: If `tokenizer` or `dataset_path` is not provided.
        """
        
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        
        self.batch_size = kwargs.get("batch_size", 16)
        self.test_size = kwargs.get("test_size", 0.2)
        self.seed = kwargs.get("seed", 42)
        self.max_length = kwargs.get("max_length", 512)
        
    # Make static?
    def clean_dataset(self, dataset):
        """
        Clean the dataset here if necessary

        Returns:
            DatasetDict: Loaded dataset.
        """
        # Define a function to filter out rows with null values
        def remove_null_rows(example):
            return all(value is not None for value in example.values())

        # Apply the filter to remove rows with nulls for each split
        filtered_dataset = {
            split: data.filter(remove_null_rows)
            for split, data in dataset.items()
        }

        return filtered_dataset

    def load_dataset(self) -> DatasetDict:
        """
        Load the dataset using Hugging Face's `load_dataset`.

        Returns:
            DatasetDict: Loaded dataset.
        """
        
        print(f"[INFO] Loading dataset from {self.dataset_path}")
        dataset = load_dataset(self.dataset_path)
        
        return self.clean_dataset(dataset=dataset)


    @abstractmethod
    def tokenize_function(self, examples):
        """
        Tokenization logic. Must be implemented by subclasses.

        Args:
            examples: Batch of dataset examples to tokenize.

        Returns:
            Tokenized examples.
        """
        
        pass

    
    def train_eval_test_split(self):
        """
        Split the dataset into train, evaluation, and test sets.

        Returns:
            tuple: Train, evaluation, and test datasets.
        """
        
        dataset = self.load_dataset()
        
        
        # Split into train (80%), eval (16%), and test (4%)
        split_dataset = dataset['train'].train_test_split(
            test_size=self.test_size, seed=self.seed
        )
        test_dataset = split_dataset["test"]

        train_test_split = split_dataset["train"].train_test_split(
            test_size=self.test_size, seed=self.seed
        )
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]

        return train_dataset, eval_dataset, test_dataset
    
    def get_datasets(self, raw_dataset):
        """
        Adjust dataset structure if needed before mapping.
        Processes raw datasets to extract user conversations and structure them
        into input-output sentence pairs.

        Args:
            raw_dataset: The raw dataset containing user conversations.

        Returns:
            Dataset: A structured dataset with input and target sentences.
        """
        return raw_dataset
    
    def preprocess(self):
        """
        Preprocess the dataset by splitting into train, eval, and test sets, and then tokenizing them.

        Returns:
            tuple: Tokenized train, eval, and test datasets.
        """

        print("[INFO] Preprocessing dataset...")
        
        train_dataset, eval_dataset, test_dataset = self.train_eval_test_split()
    

        # Tokenize the datasets using map()
        tokenized_train_dataset = self.get_datasets(train_dataset).map(self.tokenize_function, batched=True)
        tokenized_eval_dataset = self.get_datasets(eval_dataset).map(self.tokenize_function, batched=True)
        tokenized_test_dataset = self.get_datasets(test_dataset).map(self.tokenize_function, batched=True)
        
        print("Len(tokenized_train_dataset) = ", tokenized_train_dataset)
        print("Len(tokenized_eval_dataset) = ", tokenized_eval_dataset)
        print("Len(tokenized_test_dataset) = ", tokenized_test_dataset)
        
        
        # import sys
        # sys.stdout.flush()
        
        # raise("stop here")
        return tokenized_train_dataset, tokenized_eval_dataset, tokenized_test_dataset