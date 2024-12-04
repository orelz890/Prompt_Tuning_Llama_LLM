import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configuration.config import Config as conf
from huggingface_hub import login

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, get_peft_model, PeftModel, TaskType, PromptTuningInit


class ModelManager:
    def __init__(self, model_name: str, local_model_dir: str = "./local_model", device='cpu'):
        self.model_name = model_name
        self.local_model_dir = local_model_dir
        self.foundational_model = None
        self.peft_model_prompt = None
        self.tokenizer = None
        self.prompt_config = None
        self.device = device
        
        # Ensure the local model directory exists
        os.makedirs(self.local_model_dir, exist_ok=True)

    def load_model_and_tokenizer(self):
        print(f"[INFO] Loading The Foundational Model")

        if os.path.exists(os.path.join(self.local_model_dir, "config.json")):
            print(f"[INFO] Loading model and tokenizer from local directory: {self.local_model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_dir)
            self.foundational_model = AutoModelForCausalLM.from_pretrained(self.local_model_dir, trust_remote_code=True).to(self.device)
        else:
            # Hugging Face
            login(token=conf.hugging_token)
            
            print(f"[INFO] Downloading model and tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.foundational_model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
            
            # Save pretrained chat bot model locally
            self.save_model( 
                output_dir=self.local_model_dir, 
                model_type="foundational"
            )
        
        # Ensure the tokenizer has padding tokens for input and output
        if self.tokenizer.pad_token is None or self.model_manager.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or self.tokenizer.convert_tokens_to_ids("[PAD]")
            self.foundational_model.resize_token_embeddings(len(self.tokenizer))  # Resize embeddings if new tokens are added
            print(f"[INFO] Padding token set to: {self.tokenizer.pad_token}")

    #    # Set the model configuration tokens
    #     self.foundational_model.config.pad_token_id = self.tokenizer.pad_token_id
    #     self.foundational_model.config.eos_token_id = self.tokenizer.eos_token_id

    def configure_prompt_tuning(self, num_virtual_tokens):
        print(f"[INFO] Configuring Prompt Tuning with {num_virtual_tokens} virtual tokens.")
        
        # TODO - Read about it
        self.prompt_config = PromptTuningConfig( 
            task_type=TaskType.CAUSAL_LM,  # This type indicates the model will generate text. 
            prompt_tuning_init=PromptTuningInit.RANDOM,  # The added virtual tokens are initializad with random numbers
            num_virtual_tokens=num_virtual_tokens,
            tokenizer_name_or_path=self.model_name,  # The pre-trained model.
        )
        
        self.peft_model_prompt = get_peft_model(self.foundational_model, self.prompt_config)

    def load_prompt_tuned_model(self, output_dir: str):
        print(f"[INFO] Loading prompt-tuned model from {output_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        # Foundational model
        self.foundational_model = AutoModelForCausalLM.from_pretrained(
            self.local_model_dir
        ).to(self.device)
        
        # peft model
        self.peft_model_prompt = PeftModel.from_pretrained(
            self.foundational_model, 
            output_dir,
            # device_map='auto', TODO - read how to use it.
            is_trainable=False,
        ).to(self.device)


    def save_model(self, output_dir: str, model_type: str):
        """
        Save the specified model and tokenizer.
        Args:
            model_type (str): "foundational" or "peft"
        """
        
        print(f"[INFO] Saving trained model to {output_dir}")
        
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        else:
            raise ValueError("[ERROR] TOKENIZER is not initialized.")
        
        if model_type == "peft":
            if self.peft_model_prompt is None:
                raise ValueError("[ERROR] PEFT model is not initialized.")
            self.peft_model_prompt.save_pretrained(output_dir)
        elif model_type == "foundational":
            if self.foundational_model is None:
                raise ValueError("[ERROR] FOUNDATIONAL model is not initialized.")
            self.foundational_model.save_pretrained(output_dir)
        else:
            raise ValueError("[ERROR] Invalid model_type. Use 'foundational' or 'peft'.")
