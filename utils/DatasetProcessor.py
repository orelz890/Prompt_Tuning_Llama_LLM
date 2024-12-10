from abc import ABC, abstractmethod
from datasets import load_dataset, DatasetDict


class DatasetProcessor(ABC):
    """
    Abstract base class for dataset processing. 
    Requires overriding of `load_dataset`, `preprocess`, and `tokenize_function`.
    """

    def __init__(self, tokenizer, dataset_path, **kwargs):
        
        """
        Raises:
            ValueError: Must provide a tokenizer and a dataset_path
        """
        
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        
        self.batch_size = kwargs.get("batch_size", 16)
        self.test_size = kwargs.get("test_size", 0.2)
        self.seed = kwargs.get("seed", 42)
        self.max_length = kwargs.get("max_length", 512)
        

    def load_dataset(self) -> DatasetDict:
        """
        Load the dataset.
        Default using Hugging Face's `load_dataset`.
        """
        print(f"[INFO] Loading dataset from {self.dataset_path}")
        return load_dataset(self.dataset_path)


    @abstractmethod
    def tokenize_function(self, examples):
        """
        Tokenization logic. Must be implemented by subclasses.
        """
        pass

    
    def train_eval_test_split(self):
        """
        Split the dataset: split into train, eval, and test using train_test_split func.
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
            Change your dataset structure here if needed before mapping
        """
        return raw_dataset
    
    def preprocess(self):
        """
        Uses train_eval_test_split and tokenize_function to prepare and tokenize the data
        
        Returns:
            tokenized_train_dataset: DatasetDict
            tokenized_eval_dataset:  DatasetDict
            tokenized_test_dataset:  DatasetDict
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