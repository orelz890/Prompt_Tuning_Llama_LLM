import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configuration.config import Config as conf
from huggingface_hub import login

from peft import PromptTuningConfig, get_peft_model, PeftModel, TaskType, PromptTuningInit



class ModelManager:
    """
        A manager for handling foundational and prompt-tuned models, including loading, saving,
        and configuring models for prompt tuning tasks.

        Attributes:
            model_name (str): Name of the foundational model.
            local_model_dir (str): Directory for saving/loading local models.
            peft_model_prompt: Prompt-tuned model.
            tokenizer: Tokenizer for preprocessing inputs and outputs.
            prompt_config: Configuration for prompt tuning.
            device (str): The device (e.g., 'cuda', 'cpu') used for computations.
    """
    
    def __init__(self,
                tokenizer_class,
                model_loader,
                task_type,
                device: str, 
                model_name: str, 
                local_model_dir: str, 
                ):
        """
            Initialize the ModelManager.

            Args:
                model_loader (class or callable): A class or callable object for loading the model.
                                                Examples: AutoModelForCausalLM, AutoModelForSeq2SeqLM.
                task_type (TaskType): The type of task this pipeline will handle, defined by the `TaskType` enumeration.
                                    Examples: TaskType.CAUSAL_LM, TaskType.SEQ2SEQ_LM, etc.
                tokenizer_class (class): The tokenizer class associated with the model.
                                        Examples: AutoTokenizer.
                model_name (str): Name of the foundational model. (e.g., "unsloth/Llama-3.2-1B-Instruct").
                local_model_dir (str): Directory for saving/loading local models.
                device (str): The device (e.g., 'cuda', 'cpu') used for computations.
        """
        
        self.model_name = model_name
        self.local_model_dir = local_model_dir
        self.device = device
        self.tokenizer_class = tokenizer_class
        self.model_loader = model_loader
        self.task_type = task_type
        self.foundational_model = None
        self.peft_model_prompt = None
        self.tokenizer = None
        self.prompt_config = None
        
        # Ensure the local model directory exists
        os.makedirs(self.local_model_dir, exist_ok=True)
        
        print("Model_manager - model_loader: ", model_loader)

    def get_output(self, model_type, inputs, **kwargs):
        """
            Generates a response using the specified model type with the provided inputs.

            Args:
                model_type (str): The type of model to use for generating output. 
                Options:
                    - "peft": Uses the PEFT (Parameter-Efficient Fine-Tuning) model.
                    - Other: Defaults to the foundational model.
                inputs (dict): A dictionary of input parameters for the model's `generate` method. 
                            Must include all necessary fields for text generation.

            Returns:
                torch.Tensor: A tensor containing the generated token IDs, where each row corresponds
        """
                
        model = self.peft_model_prompt if model_type == "peft" else self.foundational_model
        
        # Generate model response
        return model.generate(
            **inputs,
            do_sample=kwargs.get("do_sample", True),                  # Enable sampling for varied responses
            top_p=kwargs.get("top_p", 0.9),                           # Nucleus sampling
            max_new_tokens=kwargs.get("max_new_tokens", 100),         # Maximum new tokens to generate
            repetition_penalty=kwargs.get("repetition_penalty", 1.2), # Penalize repetition
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def load_model_and_tokenizer(self):
        """
            Load the foundational model and tokenizer, either from a local directory or the Hugging Face Hub.

            Behavior:
                - If the model exists in the specified local directory (`self.local_model_dir`), it loads the model and tokenizer from there.
                - If the model is not found locally, it downloads the model and tokenizer from the Hugging Face Hub using the provided model name (`self.model_name`).
                - After downloading, the model is saved locally in the specified directory (`self.local_model_dir`).
                - Ensures the tokenizer has a valid padding token and adjusts the model's token embeddings accordingly.
        """
        
        if os.path.exists(os.path.join(self.local_model_dir, self.model_name)):
            # Load from local dir is exists
            print(f"[INFO] Loading model and tokenizer from local directory: {self.local_model_dir}")
            self.tokenizer = self.tokenizer_class.from_pretrained(self.local_model_dir)
            self.foundational_model = self.model_loader.from_pretrained(self.local_model_dir).to(self.device)
        
        else:
            # Download from Hugging Face
            login(token=conf.API_KEYS.get("hugging_token"))
            
            print(f"[INFO] Downloading model and tokenizer: {self.model_name}")
            self.tokenizer = self.tokenizer_class.from_pretrained(self.model_name)
            self.foundational_model = self.model_loader.from_pretrained(self.model_name).to(self.device)
            
            # Save pretrained chat bot model locally
            self.save_model( 
                output_dir=self.local_model_dir, 
                model_type="foundational"
            )
        
        # TODO - Check if setting the pad id to the eos is fine?
        # Ensure the tokenizer has padding tokens for input and output
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
            self.foundational_model.resize_token_embeddings(len(self.tokenizer))  # Resize embeddings if new tokens are added
            print(f"[INFO] Padding token set to: {self.tokenizer.pad_token}. token_id: {self.tokenizer.pad_token_id}")

    def configure_prompt_tuning(self, num_virtual_tokens, prompt_tuning_init_text):
        """
            Configures prompt tuning for the foundational model using a specified number of virtual tokens.

            This method sets up a `PromptTuningConfig` to define how prompt tuning is applied to the model. 
            It also initializes a PEFT (Parameter-Efficient Fine-Tuning) model using the foundational model 
            and the prompt tuning configuration.

            Args:
                num_virtual_tokens (int): The number of virtual tokens to prepend for prompt tuning. 
                                        These tokens act as learnable parameters that influence the model's output.

            Behavior:
                - Prints an informational message about the number of virtual tokens being configured.
                - Sets up a `PromptTuningConfig` object with the following details:
                    - `task_type`: The type of task (e.g., causal language modeling).
                    - `prompt_tuning_init`: Specifies how the virtual tokens are initialized (in this case, from text).
                    - `num_virtual_tokens`: Number of virtual tokens to add.
                    - `prompt_tuning_init_text`: A textual initialization for the virtual tokens, providing the model 
                                                with context about its behavior and purpose.
                    - `tokenizer_name_or_path`: Path or name of the tokenizer associated with the foundational model.
                - Initializes a PEFT model (`self.peft_model_prompt`) by applying the prompt tuning configuration 
                to the foundational model.
        """
        print(f"[INFO] Configuring Prompt Tuning with {num_virtual_tokens} virtual tokens.")
        
        
        # TODO - Read about it
        self.prompt_config = PromptTuningConfig(
            task_type=self.task_type or TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init_text=prompt_tuning_init_text,
            tokenizer_name_or_path=self.model_name,  # The pre-trained model.
        )
        
        self.peft_model_prompt = get_peft_model(self.foundational_model, self.prompt_config)

    def load_prompt_tuned_model(self, output_dir: str):
        """
            Loads a prompt-tuned model and its tokenizer from the specified directory.

            This method initializes both the foundational model and the prompt-tuned model using the
            configurations saved in the specified `output_dir`. It ensures that the tokenizer and
            models are ready for inference or further evaluation.

            Args:
                output_dir (str): The directory where the prompt-tuned model and tokenizer are stored.

        """
        
        print(f"[INFO] Loading prompt-tuned model from {output_dir}")
        self.tokenizer = self.tokenizer_class.from_pretrained(output_dir)
        
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
