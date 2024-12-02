import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configuration.config import Config as conf
from huggingface_hub import login

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, get_peft_model, PeftModel

class ModelManager:
    def __init__(self, model_name: str, local_model_dir: str = "./local_model", device='cpu'):
        self.model_name = model_name
        self.local_model_dir = local_model_dir
        self.model = None
        self.tokenizer = None
        self.device = device
        
        # Ensure the local model directory exists
        os.makedirs(self.local_model_dir, exist_ok=True)

    def load_model_and_tokenizer(self):
        if os.path.exists(os.path.join(self.local_model_dir, "config.json")):
            print(f"[INFO] Loading model and tokenizer from local directory: {self.local_model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(self.local_model_dir).to(self.device)
        else:
            # Hugging Face
            login(token=conf.hugging_token)
            
            print(f"[INFO] Downloading model and tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            
            # Save the model locally
            self.tokenizer.save_pretrained(self.local_model_dir)
            self.model.save_pretrained(self.local_model_dir)
        
        # Ensure the tokenizer has padding tokens for input and output
        if self.tokenizer.pad_token is None or self.model_manager.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or self.tokenizer.convert_tokens_to_ids("[PAD]")
            self.model.resize_token_embeddings(len(self.tokenizer))  # Resize embeddings if new tokens are added
            print(f"[INFO] Padding token set to: {self.tokenizer.pad_token}")

       # Set the model configuration tokens
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

    def configure_prompt_tuning(self, num_virtual_tokens):
        print(f"[INFO] Configuring Prompt Tuning with {num_virtual_tokens} virtual tokens.")
        prompt_config = PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=num_virtual_tokens)
        self.model = get_peft_model(self.model, prompt_config)

    def load_prompt_tuned_model(self, output_dir: str):
        print(f"[INFO] Loading prompt-tuned model from {output_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(output_dir)
        base_model = AutoModelForCausalLM.from_pretrained(self.local_model_dir).to(self.device)
        self.model = PeftModel.from_pretrained(base_model, output_dir).to(self.device)
