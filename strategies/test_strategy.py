from datasets import load_dataset
from evaluate import load
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.base_strategy import BasePipelineStrategy
from managers.model_manager import ModelManager
from utils.DatasetProcessor import DatasetProcessor
from utils.CustomDataCollatorSameSize import CustomDataCollatorSameSize
from torch.nn.functional import cross_entropy

# Fine-Tuning Strategy
class TestStrategy(BasePipelineStrategy):
    def __init__(self, model_manager: ModelManager, dataset_path: str, dataset_processor: DatasetProcessor, **kwargs):
        
        self.model_manager = model_manager

        dp = dataset_processor(
            tokenizer=model_manager.tokenizer,
            dataset_path=dataset_path,
            **kwargs
        )
        
        # Print dataset setup info
        dp.print_args()
        
        _, _, test_dataset = dp.train_eval_test_split()
        input, label = dp.get_input_label_columns_names()
        
        # Rename columns
        self.dataset = test_dataset.map(
            lambda x: {"data": x[input], "label": x[label]},
            remove_columns=[input, label],
            batched=True
        )

        print(self.dataset)
        
        _, _, test_dataset2 = dp.train_eval_test_split()
        
        data_collator = CustomDataCollatorSameSize(
            tokenizer=model_manager.tokenizer,
            device=model_manager.device
        )
        
        dp.proc
        
        self.data_loader = DataLoader(
            test_dataset2,  # Your Dataset object
            batch_size=1,  # Adjust based on memory
            collate_fn=data_collator  # Use your DataCollator
        )
    
    # Function to normalize text
    @staticmethod
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text
    
    def execute(self):
        print("[INFO] Start Testing.")
        
        self.model_manager.foundational_model.eval()
        self.model_manager.peft_model_prompt.eval()

        # Bleu
        # self.calc_bleu_score()
        
        # Perplexity
        self.calc_perplexity()

    def calc_perplexity(self):
        # Iterate over dataset
        for batch in self.data_loader:
            with torch.no_grad():
                print(batch, type(batch))
                raise("stop")

    def calc_bleu_score(self):
        # Load dataset, model, tokenizer, and BLEU metric
        bleu = load("bleu")

        foundational_bleu_scores = []
        peft_bleu_scores = []

        # Iterate over dataset
        for batch in self.dataset:
            with torch.no_grad():
                print(batch)
                
                user_input = batch['data']
                label = batch['label']
                
                # Format the input as an instruct-style message
                message = [{"role": "user", "content": user_input}]
                inputs = self.model_manager.tokenizer.apply_chat_template(message, add_generation_prompt=True, return_dict=True, return_tensors="pt")
                inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}

                # Generate model response
                f_outputs = self.model_manager.get_output(
                    inputs=inputs,
                    model_type="foundational",
                )
                
                p_outputs = self.model_manager.get_output(
                    inputs=inputs,
                    model_type="peft",
                )
                
                f_response = self.model_manager.tokenizer.decode(f_outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
                f_response = TestStrategy.normalize_text(f_response)
                p_response = self.model_manager.tokenizer.decode(p_outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
                p_response = TestStrategy.normalize_text(p_response)
                
                # print("[INPUT]: ", batch['data'])
                # print("[Foundational MODEL]: ", f_response)
                # print("[Prompt Tuning MODEL]: ", p_response)
                # print("[label]: ", label)
                
                # BLEU
                f_bleu_results = bleu.compute(predictions=[f_response], references=[label], smooth=True)
                foundational_bleu_scores.append(f_bleu_results["bleu"])
                
                p_bleu_results = bleu.compute(predictions=[p_response], references=[label], smooth=True)
                peft_bleu_scores.append(p_bleu_results["bleu"])

        # # Final results
        print(f"Foundational Model Average BLEU Score: {sum(foundational_bleu_scores) / len(foundational_bleu_scores)}")
        print(f"Prompt Tuning Model Average BLEU Score: {sum(peft_bleu_scores) / len(peft_bleu_scores)}")
