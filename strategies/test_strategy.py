from datasets import load_dataset
from evaluate import load
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from collections import defaultdict

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
                
        data_collator = CustomDataCollatorSameSize(
            tokenizer=model_manager.tokenizer,
            device=model_manager.device
        )
        
        _, _, tokenized_test_dataset = dp.preprocess()
        
        self.data_loader = DataLoader(
            tokenized_test_dataset,  # Your Dataset object
            batch_size=1,  # Adjust based on memory
            collate_fn=data_collator  # Use your DataCollator
        )

    def execute(self):
        print("[INFO] Start Testing.")
        
        self.model_manager.foundational_model.eval()
        self.model_manager.peft_model_prompt.eval()

        # Bleu
        
        TestStrategy.print_scores(self.calc_scores())
        
        # Perplexity
        # self.calc_perplexity()

    # Function to normalize text
    @staticmethod
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text
    
    staticmethod
    def calc_loss(model, inputs, labels):
        logits = model(
            **inputs
        ).logits
        
        num_virtual_tokens = logits.shape[1] - labels.shape[1]
        logits = logits[:, num_virtual_tokens:, :].contiguous()

        # print(foundational_outputs)
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),  # Flatten logits
            labels.view(-1),                   # Flatten labels
            ignore_index=-100                  # Ignore padding tokens
        )
        return loss

    def calc_perplexity(self):
        # Iterate over dataset
        f_perplexities = []
        p_perplexities = []
        
        for batch in self.data_loader:
            with torch.no_grad():
                batch = {k: v.to(self.model_manager.device) for k, v in batch.items()}
 
                f_loss = TestStrategy.calc_loss(self.model_manager.foundational_model, batch, batch['labels'])
                p_loss = TestStrategy.calc_loss(self.model_manager.peft_model_prompt, batch, batch['labels'])

                f_perplexities.append(torch.exp(f_loss).item())
                p_perplexities.append(torch.exp(p_loss).item())

        f_average_perplexity = sum(f_perplexities) / len(f_perplexities)
        p_average_perplexity = sum(p_perplexities) / len(p_perplexities)

        print(f"Foundational Average Perplexity: {f_average_perplexity}")
        print(f"Prompt Tuning Average Perplexity: {p_average_perplexity}")
        
        winner = "Prompt Tuning" if p_average_perplexity < f_average_perplexity else "Llama"
        print(f"Winner: ", winner)
        return winner
        
    def calc_scores(self):
        # Measures Lexical Variation and Semantics - evaluate meaning
        bleu = load("bleu")
        
        # Measures n-gram overlap between the generated text and the reference text, focusing on recall.
        rouge = load("rouge")
        
        # Considers synonyms, stemming and word order
        meteor = load("meteor")

        # Pretrained BERT model to evaluate the semantic similarity of two texts
        bertscore = load("bertscore")
        
        
        scores = defaultdict(lambda: ([],[]))
        
        # Iterate over dataset
        for batch in self.dataset:
            with torch.no_grad():                
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
                p_response = self.model_manager.tokenizer.decode(p_outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
                f_response = TestStrategy.normalize_text(f_response)
                p_response = TestStrategy.normalize_text(p_response)
                
                # print("[INPUT]: ", batch['data'])
                # print("[Foundational MODEL]: ", f_response)
                # print("[Prompt Tuning MODEL]: ", p_response)
                # print("[label]: ", label)
                
                # BLEU
                # scores['bleu'][0].append(bleu.compute(predictions=[f_response], references=[label], smooth=True))
                # scores['bleu'][1].append(bleu.compute(predictions=[p_response], references=[label], smooth=True))
                
                # scores['rouge'][0].append(rouge.compute(predictions=[f_response], references=[label]))
                # scores['rouge'][1].append(rouge.compute(predictions=[p_response], references=[label]))
                
                # scores['meteor'][0].append(meteor.compute(predictions=[f_response], references=[label]))
                # scores['meteor'][1].append(meteor.compute(predictions=[p_response], references=[label]))

                scores['bertscore'][0].append(bertscore.compute(predictions=[f_response], references=[label], lang="en"))
                scores['bertscore'][1].append(bertscore.compute(predictions=[p_response], references=[label], lang="en"))
        return scores
    
    @staticmethod
    def compare_rouge_scores(f_scores, p_scores):
        """
        Compares ROUGE scores for two models and determines which is better overall.
        Returns the name of the better model and the comparison for each score.
        """
        metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        comparison = {}

        # Compare metrics for each score
        for metric in metrics:
            f_average = sum([x[metric] for x in f_scores]) / len(f_scores)
            p_average = sum([x[metric] for x in p_scores]) / len(p_scores)
            
            if p_average > f_average:
                comparison[metric] = 'PT'
            else:
                comparison[metric] = 'Llama'

        # Count the "wins" for each model
        model1_wins = sum(1 for v in comparison.values() if v == 'PT')
        model2_wins = sum(1 for v in comparison.values() if v == 'Llama')

        # Determine the better model overall
        return 'Prompt Tuning' if model1_wins > model2_wins else 'Llama'

    def print_scores(scores):
        
        for k, v in scores.items():
            f_scores = v[0]
            p_scores = v[1]
            
            print(f_scores, type(f_scores))
            if k == 'bleu':
                f_scores = [x['bleu'] for x in f_scores]
                p_scores = [x['bleu'] for x in p_scores]
                
                f_average = sum(f_scores) / len(f_scores)
                p_average = sum(p_scores) / len(p_scores)
                
                print(f"Foundational Model Average BLEU Score: {f_average}")
                print(f"Prompt Tuning Model Average BLEU Score: {p_average}")
                print("Better BLEU Score: ", "Prompt Tuning" if p_average > f_average else "Llama")
            elif k == 'rouge':
                print("Better ROUGE Score: ", TestStrategy.compare_rouge_scores(f_scores, p_scores))
            elif k == 'meteor':
                f_average = sum([x['meteor'] for x in f_scores]) / len(f_scores)
                p_average = sum([x['meteor'] for x in p_scores]) / len(p_scores)
                
                print(f"Foundational Model Average METEOR Score: {f_average}")
                print(f"Prompt Tuning Model Average METEOR Score: {p_average}")
                print("Better METEOR Score: ","Prompt Tuning" if p_average > f_average else "Llama")
            elif k == 'bertscore':
                f_average = sum([x['f1'][0] for x in f_scores]) / len(f_scores)
                p_average = sum([x['f1'][0] for x in p_scores]) / len(p_scores)
                
                print(f"Foundational Model Average BERT Score: {f_average}")
                print(f"Prompt Tuning Model Average BERT Score: {p_average}")
                print("Better METEOR Score: ","Prompt Tuning" if p_average > f_average else "Llama")
            