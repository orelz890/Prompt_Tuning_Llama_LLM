import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configuration.config import Config as conf
from huggingface_hub import login

from transformers import AutoModelForCausalLM, AutoTokenizer, BlenderbotForConditionalGeneration, AutoModelForSeq2SeqLM
from peft import PromptTuningConfig, get_peft_model, PeftModel, TaskType, PromptTuningInit, LoraConfig



class ModelManager:
    """
    A manager for handling foundational and prompt-tuned models, including loading, saving,
    and configuring models for prompt tuning tasks.

    Attributes:
        model_name (str): Name of the foundational model.
        local_model_dir (str): Directory for saving/loading local models.
        foundational_model: The base model used for tasks.
        peft_model_prompt: Prompt-tuned model.
        tokenizer: Tokenizer for preprocessing inputs and outputs.
        prompt_config: Configuration for prompt tuning.
        device (str): The device (e.g., 'cuda', 'cpu') used for computations.
    """
    
    def __init__(self,
                auto_tokenizer,
                model_loader,
                task_type,
                model_name: str, 
                local_model_dir: str = "./local_model", 
                device='cpu', 
                ):
        """
        Initialize the ModelManager.

        Args:
            model_name (str): Name of the foundational model.
            local_model_dir (str): Directory for saving/loading local models. Default is "./local_model".
            device (str): The device for computation ('cpu' or 'cuda'). Default is 'cpu'.
        """
        
        self.model_name = model_name
        self.local_model_dir = local_model_dir
        self.device = device
        self.auto_tokenizer = auto_tokenizer
        self.model_loader = model_loader
        self.task_type = task_type
        
        print("Model_manager - model_loader: ", model_loader)
        
        self.foundational_model = None
        self.peft_model_prompt = None
        self.tokenizer = None
        self.prompt_config = None
        
        # Ensure the local model directory exists
        os.makedirs(self.local_model_dir, exist_ok=True)

    def get_output(self, model_type, inputs):
        model = self.peft_model_prompt if model_type == "peft" else self.foundational_model
        
        # Generate model response
        return model.generate(
            **inputs,
            do_sample=True,             # Enable sampling for varied responses
            top_p=0.9,                  # Nucleus sampling
            max_new_tokens=100,         # Maximum new tokens to generate
            repetition_penalty=1.2,     # Penalize repetition
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def load_model_and_tokenizer(self):
        """
        Load the foundational model and tokenizer, either from a local directory or the Hugging Face Hub.
        """
        
        print(f"[INFO] Loading The Foundational Model: ", os.path.join(self.local_model_dir, self.model_name))

        if os.path.exists(os.path.join(self.local_model_dir, self.model_name)):
            # Load from local dir is exists
            print(f"[INFO] Loading model and tokenizer from local directory: {self.local_model_dir}")
            self.tokenizer = self.auto_tokenizer.from_pretrained(self.local_model_dir)
            self.foundational_model = self.model_loader.from_pretrained(self.local_model_dir).to(self.device)
        
        else:
            # Download from Hugging Face
            login(token=conf.API_KEYS.get("hugging_token"))
            
            print(f"[INFO] Downloading model and tokenizer: {self.model_name}")
            self.tokenizer = self.auto_tokenizer.from_pretrained(self.model_name)
            self.foundational_model = self.model_loader.from_pretrained(self.model_name).to(self.device)
            
            # Save pretrained chat bot model locally
            self.save_model( 
                output_dir=self.local_model_dir, 
                model_type="foundational"
            )
        
        # TODO - Check if setting the pad id to the eos is fine?
        # Ensure the tokenizer has padding tokens for input and output
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or self.tokenizer.convert_tokens_to_ids("[PAD]")
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
            self.foundational_model.resize_token_embeddings(len(self.tokenizer))  # Resize embeddings if new tokens are added
            print(f"[INFO] Padding token set to: {self.tokenizer.pad_token}. token_id: {self.tokenizer.pad_token_id}")

    def configure_prompt_tuning(self, num_virtual_tokens):
        """
        Configure the foundational model for prompt tuning.

        Args:
            num_virtual_tokens (int): Number of virtual tokens for prompt tuning.
        """
        
        print(f"[INFO] Configuring Prompt Tuning with {num_virtual_tokens} virtual tokens.")
        
        prompt_tuning_init_text="You are an AI designed to pass the Turing Test by mimicking human communication. Occasionally make errors like typos or hesitations. Don't know everything - use 'I'm not sure' or equivalent and speculate when needed. Stay context-aware, clear and personable, avoiding robotic precision."
        
        # TODO - Read about it
        self.prompt_config = PromptTuningConfig(
            task_type=self.task_type or TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,  # The added virtual tokens are initialized with random numbers
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init_text=prompt_tuning_init_text,
            tokenizer_name_or_path=self.model_name,  # The pre-trained model.
        )
        
        self.peft_model_prompt = get_peft_model(self.foundational_model, self.prompt_config)

    def load_prompt_tuned_model(self, output_dir: str):
        """
        Load a pretrained prompt-tuned model from a specified directory.

        Args:
            output_dir (str): Directory containing the prompt-tuned model.
        """
        
        print(output_dir, self.auto_tokenizer)
        print(f"[INFO] Loading prompt-tuned model from {output_dir}")
        self.tokenizer = self.auto_tokenizer.from_pretrained(output_dir)
        
        # Foundational model
        self.foundational_model = self.model_loader.from_pretrained(
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
            output_dir (str): Directory to save the model and tokenizer.
            model_type (str): Type of model to save ('foundational' or 'peft').

        Raises:
            ValueError: If the specified model type is invalid or not initialized.
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

