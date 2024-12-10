import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from datasets import load_dataset, DatasetDict, Dataset


from utils.DatasetProcessor import DatasetProcessor

class GooglePersonaDatasetProcessor(DatasetProcessor):

    def train_eval_test_split(self):
        dataset = self.load_dataset()
        print(len(dataset["train"]))
        return dataset["train"], dataset["validation"], dataset["test"]
        
    def tokenize_function(self, examples):
        """
        Tokenization logic for a specific dataset.
        """

        # print("conv: ", examples["input_sentences"])
        

        user1_sentences = examples["input_sentences"]
        user2_sentences = examples["target_sentences"]
        
        # print("conv: ", type(examples["Best Generated Conversation"][0]))

        # print("len(user1_sentences): ", len(user1_sentences))
        # print("len(user2_sentences): ", len(user2_sentences))

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
        
        
        # print("Length of tokenized inputs:", len(inputs["input_ids"]))
        # print("Length of tokenized labels:", len(labels))
        
        
        # import sys
        # sys.stdout.flush()
        
        # raise("stop here")
    
        return inputs

    
    def get_datasets(self, raw_dataset):
        
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
            
    