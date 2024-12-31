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

    def get_input_label_columns_names(self):
        return 'Questions', 'Answers'

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
        dataset = DatasetProcessor.clean_dataset(dataset=dataset)
        
        return dataset
