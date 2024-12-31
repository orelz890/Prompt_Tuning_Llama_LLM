from abc import ABC, abstractmethod
from datasets import load_dataset, DatasetDict


class DatasetProcessor(ABC):
    """
    Abstract base class for dataset processing.
    Requires overriding of `load_dataset`, `get_input_label_columns_names`.

    Attributes:
        tokenizer: Tokenizer for processing the dataset.
        dataset_path (str): Path to the dataset.
        batch_size (int): Batch size for tokenizing the dataset. Default is 16.
        test_size (float): Fraction of the dataset reserved for testing. Default is 0.2.
        seed (int): Random seed for reproducibility. Default is 42.
        max_length (int): Maximum token length. Default is 128.
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
                - max_length (int): Maximum token length. Default is 128.

        Raises:
            ValueError: If `tokenizer` or `dataset_path` is not provided.
        """
        
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.batch_size = kwargs.get("batch_size", 16)
        self.test_size = kwargs.get("test_size", 0.2)
        self.seed = kwargs.get("seed", 42)
        self.max_length = kwargs.get("max_length", 128)
        
    @staticmethod
    def clean_dataset(dataset):
        """
        Clean the dataset here if necessary

        Args:
            dataset (DatasetDict): The dataset to be cleaned, with potential splits (e.g., "train", "test").

        Returns:
            DatasetDict: The cleaned dataset
        """
        # Define a function to filter out rows with null values
        def remove_null_rows(example):
            return all(value is not None for value in example.values())

        # Apply the filter to remove rows with nulls for each split
        for split in dataset.keys():
            dataset[split] = dataset[split].filter(remove_null_rows)

        return dataset
    
    def apply_chat_template(self, input_sentence, response):
        """
            Formats input and response into a structured chat template for the model.

            Args:
                input_sentence (str): The user's input or query.
                response (str): The assistant's response.

            Returns:
                dict: A dictionary containing the formatted prompt, ready for tokenization or generation.

            Behavior:
                - Constructs a chat template with roles (`user`, `assistant`) and their respective content.
                - Uses the tokenizer to apply the chat template, optionally including a generation prompt.

        """
        messages = [
            {"role": "user", "content": input_sentence},
            {"role": "assistant", "content": response}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt}
    
    def apply_chat(self, example):
        """
            Applies the chat template to a given example by formatting input and response.

            Args:
                example (dict): A dictionary containing the input sentence and corresponding response.

            Returns:
                dict: A dictionary with the formatted chat prompt.
        """
        input_column, label_column = self.get_input_label_columns_names()
        return self.apply_chat_template(input_sentence=example[input_column], response=example[label_column])

    @abstractmethod
    def load_dataset(self) -> DatasetDict:
        """
            Load your dataset here.

            Returns:
                DatasetDict: Loaded dataset.
        """
        pass
    
    @abstractmethod
    def get_input_label_columns_names(self):
        """
            Abstract method to define the input and label column names.

            This method must be implemented in subclasses to specify the names of the columns
            used for the input sentences and corresponding labels in the dataset.

            Raises:
                NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("The method 'get_input_label_columns_names' must be implemented in a subclass.")

    def tokenize_function(self, examples):
        """
            Tokenizes the input examples and prepares them for model training.

            Args:
                examples (dict): A dictionary containing the input data with a 'prompt' key.

            Returns:
                dict: A dictionary containing tokenized inputs and processed labels.

            Behavior:
                - Tokenizes the 'prompt' field from the input examples with padding, truncation, and a maximum length of 128 tokens.
                - Sets padding token labels to -100 to ensure they are ignored during loss calculation.
        """
        
        tokens = self.tokenizer(examples['prompt'], padding="max_length", truncation=True, max_length=self.max_length)
        # Set padding token labels to -100 to ignore them in loss calculation
        tokens['labels'] = [
            -100 if token == self.tokenizer.pad_token_id else token for token in tokens['input_ids']
        ]
        
        return tokens

    def train_eval_test_split(self):
        """
            Splits the dataset into train, evaluation, and test sets.

            This method first loads the dataset, then performs successive splits to create 
            train, evaluation, and test datasets. The proportions for the splits are determined 
            by `self.test_size`. If self.test_size == 0.2, split into train (80%), eval (16%), and test (4%).

            Returns:
                tuple: A tuple containing three datasets:
                    - train_dataset: The training dataset (largest portion).
                    - eval_dataset: The evaluation dataset (used for validation).
                    - test_dataset: The test dataset (used for final evaluation).
        """
        
        dataset = self.load_dataset()
        
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
            Preprocesses the dataset by splitting it into train, eval, and test sets, 
            applying chat formatting, and tokenizing the data.

            Returns:
                tuple: A tuple containing tokenized datasets:
                    - tokenized_train_dataset: The tokenized training dataset.
                    - tokenized_eval_dataset: The tokenized evaluation dataset.
                    - tokenized_test_dataset: The tokenized test dataset.
        """

        print("[INFO] Preprocessing dataset...")

        train_dataset, eval_dataset, test_dataset = self.train_eval_test_split()

        # Reformat dataset to fit 'instruct' models dataset
        chat_format_train_dataset = self.get_datasets(train_dataset).map(self.apply_chat)
        chat_format_eval_dataset = self.get_datasets(eval_dataset).map(self.apply_chat)
        chat_format_test_dataset = self.get_datasets(test_dataset).map(self.apply_chat)

        # Tokenize the datasets using map()
        tokenized_train_dataset = chat_format_train_dataset.map(self.tokenize_function, batched=True)
        tokenized_eval_dataset = chat_format_eval_dataset.map(self.tokenize_function, batched=True)
        tokenized_test_dataset = chat_format_test_dataset.map(self.tokenize_function, batched=True)
        
        # Remove unused columns to keep only model-required fields after tokenization.
        input_column, label_column = self.get_input_label_columns_names()
        tokenized_train_dataset = tokenized_train_dataset.remove_columns([input_column, label_column, 'prompt'])    
        tokenized_eval_dataset = tokenized_eval_dataset.remove_columns([input_column, label_column, 'prompt'])
        tokenized_test_dataset = tokenized_test_dataset.remove_columns([input_column, label_column, 'prompt'])

        print("Len(tokenized_train_dataset) = ", tokenized_train_dataset)
        print("Len(tokenized_eval_dataset) = ", tokenized_eval_dataset)
        print("Len(tokenized_test_dataset) = ", tokenized_test_dataset)
        
        return tokenized_train_dataset, tokenized_eval_dataset, tokenized_test_dataset