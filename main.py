import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"

import gc

gc.collect()

from configuration.config import Config as conf
from huggingface_hub import login
from pipeline.prompt_tuning_pipeline import PromptTuningPipeline

import torch

device = 'cpu'

if torch.cuda.is_available():
    # Clear cached data
    torch.cuda.empty_cache()
    
    # Force PyTorch to release cached memory
    torch.cuda.memory_stats()

    # torch.cuda.set_per_process_memory_fraction(0.2)
    print(torch.cuda.memory_summary(device="cuda"))
    device = 'cuda'
    
    
# Define the base output directory
base_output_dir = "./pretrained"
base_local_model_dir = "./local_model"

# model_name = "meta-llama/Llama-3.1-8b-instruct"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"

dataset_path="Aviman1/Bot_Human_Prompt_Tuning_Dataset"


def main():
    
    print("Using Device: " + device)
            
    local_model_dir = os.path.join(base_local_model_dir, model_name.lower())
    
    actions = {'infer': 1, 'train': 2}
    
    while True:
        
        try:
            print("Actions: infer: 1, train: 2")
            action = int(input("[Enter Action Number]: "))
            
            if action > 2:
                raise ValueError("Invalid action number. Please enter 1 or 2.")
            
            subfolder_name = input("[Enter model name]: ")
            output_dir = os.path.join(base_output_dir, model_name.lower(), subfolder_name.lower())
            
            if action == actions['train']:
                # Check if the name already exist
                if os.path.isdir(output_dir):
                    print(f"The directory {output_dir} already exists.")
                    continue

            pipeline: PromptTuningPipeline = PromptTuningPipeline(
                model_name=model_name,
                dataset_path=dataset_path,
                output_dir=output_dir,
                local_model_dir=local_model_dir,
                device=device
            )
            
            if action == actions['train']:
                pipeline.train(epochs=1)
            else:
                pipeline.infer()
        
        except ValueError as e:
            print(e)
            continue
    

if __name__ == "__main__":
    main()


""" 
    TODO:
    
    1. [-] Understand why it always generate the max amount - padding data? warning?
    2. [v] Probably related, why we get the warning "Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation."
    3. [v] Fix GPU out of memory.
    4. [-] Talk with Avi about the dataset - get a better one.
    5. [-] Complete the other strategies.
    6. [-] Test the model before training.
    7. [-] Test the model after small amount of epochs training.
    8. [-] Ask Amos if we want a lot of epochs - intentional overfiting?
    9. [v] Find a smaller chat bot model.
""" 
