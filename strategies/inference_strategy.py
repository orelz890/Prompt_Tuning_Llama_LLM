from configuration.config import Config

conf = Config()

import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.base_strategy import BasePipelineStrategy
from managers.model_manager import ModelManager

# Suppress the position IDs warning
import warnings

warnings.filterwarnings(
    "ignore",
    message="Position ids are not supported for parameter efficient tuning. Ignoring position ids."
)

# Inference Strategy
class InferenceStrategy(BasePipelineStrategy):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.device = model_manager.device
        
    def execute(self, model_type: str, **kwargs):
        """
        Execute the inference process in an interactive chat mode.

        Args:
            model_type (str): Specifies the type of model to use ('foundational' or 'peft').
            
            **kwargs: Additional inference configurations, including:
            
            TOKENIZER:
            - max_length (int): Maximum length for tokenization. Default is 512.
            
            INFERRING:
            - temperature (float): Sampling temperature for inference. Default is 1.
            - top_p (float): Probability threshold for nucleus sampling. Default is 0.9.
            - top_k (int): Number of highest probability vocabulary tokens to keep for top-k filtering. Default is 3.
            - max_tokens_length (int): Maximum length of generated tokens. Default is 20.
            - min_tokens_length (int): Minimum length of generated tokens. Default is 1.
            - num_beams (int): Number of beams for beam search. Default is 3.
            - length_penalty (float): Length penalty for beam search. Default is 5.0.
            - repetition_penalty (float): Penalty for repeated sequences. Default is 1.5.
            - early_stopping (bool): Whether to stop early in beam search. Default is True.
            - do_sample (bool): Whether to sample from the distribution. Default is True.
        """
        
        print("[INFO] Entering interactive chat mode. Type 'exit' to quit.")

        with torch.no_grad():
            while True:
                user_input = input("\n[USER]: ")
                if user_input.lower() in {"exit", "quit"}:
                    print("[INFO] Exiting chat.")
                    break

                # Tokenize user input with attention mask
                inputs = self.model_manager.tokenizer(
                    user_input,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=kwargs.get("max_length")  # Cap input length
                ).to(self.device)


                # Generate output
                outputs = self.get_outputs(
                    inputs=inputs,
                    model_type=model_type,
                    **kwargs
                )
                
                print(user_input)
                
                # Print info about the model inputs and outputs
                self.print_info(inputs=inputs, outputs=outputs)

                # Decode and print the model's response
                response = self.model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"[MODEL]: {response}\n")
                
    def get_outputs(self, inputs, model_type: str, **kwargs):
        
        """
            Generate outputs from the model based on the provided inputs and configurations.
            
            Args:
                model_type (str): Specifies the type of model to use ('foundational' or 'peft').
                **kwargs: Additional inference configurations
            
            Returns:
                torch.Tensor: The generated outputs from the model.
        """
        # TODO - handle errors
        if model_type == "foundational":
            model = self.model_manager.foundational_model
        else:
            model = self.model_manager.peft_model_prompt
        
        outputs = model.generate(
            input_ids= inputs["input_ids"],
            attention_mask= inputs["attention_mask"],
            do_sample= kwargs.get("do_sample"),
            # max_new_tokens= kwargs.get("max_new_tokens"),
            max_length=kwargs.get("max_tokens_length"),
            min_length=kwargs.get("min_tokens_length"),
            length_penalty=kwargs.get("length_penalty"),
            num_beams=kwargs.get("num_beams"),  # Use beam search
            temperature= kwargs.get("temperature"),
            top_p= kwargs.get("top_p"),
            top_k= kwargs.get("top_k"),
            repetition_penalty= kwargs.get("repetition_penalty"),  # Avoid repetition.
            early_stopping= kwargs.get("early_stopping"),  # The model can stop before reach the max_length
            eos_token_id= self.model_manager.tokenizer.eos_token_id,
            pad_token_id=self.model_manager.tokenizer.pad_token_id,
        )
        return outputs
    
    def print_info(self, inputs, outputs):
        """
        Print detailed information about inputs and outputs during inference.

        Args:
            inputs (dict): Tokenized inputs for the model.
            outputs (torch.Tensor): Generated outputs from the model.
        """
        
        # Convert input IDs to tokens for the entire batch
        print(
            [
                self.model_manager.tokenizer.convert_ids_to_tokens(id)
                for input_ids in inputs["input_ids"]
                for id in input_ids.tolist()
            ],
        )

        # Convert output IDs to tokens
        print(
            [self.model_manager.tokenizer.convert_ids_to_tokens(id) for id in outputs[0].tolist()],
        )

        # Count total tokens in the generated output
        total_tokens = outputs.size(1)  # Outputs is of shape (batch_size, seq_length)

        # Count input tokens (optional, if you want new tokens only)
        input_token_count = inputs["input_ids"].size(1)

        # Count only newly generated tokens
        new_tokens_count = total_tokens - input_token_count

        print(f"Total tokens in output: {total_tokens}")
        print(f"Input tokens: {input_token_count}")
        print(f"Newly generated tokens: {new_tokens_count}")