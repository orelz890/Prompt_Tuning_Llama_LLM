from configuration.config import Config as conf
from huggingface_hub import login
from pipeline.prompt_tuning_pipeline import PromptTuningPipeline

import torch
import os

device = 'cpu'

# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     torch.cuda.set_per_process_memory_fraction(0.2)
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#     print(torch.cuda.memory_summary(device="cuda"))
#     device = 'cuda'

# Define the base output directory
# base_output_dir = "./prompt_tuning_results"
base_output_dir = "./pretrained"

def main():
        
    subfolder_name = input("[Enter model name]: ")
    output_dir = os.path.join(base_output_dir, subfolder_name.lower())
    # output_dir = base_output_dir
    
    login(token=conf.hugging_token)

    pipeline: PromptTuningPipeline = PromptTuningPipeline(
        model_name="meta-llama/Llama-3.1-8b-instruct",
        dataset_path="Aviman1/Bot_Human_Prompt_Tuning_Dataset",
        output_dir=output_dir,
        device=device
    )
    # pipeline.train(epochs=3)

    pipeline.infer()

if __name__ == "__main__":
    main()


""" 
    TODO:
    
    1. Understand why it always generate the max amount - padding data? warning?
    2. Probably related, why we get the warning "Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation."
    3. Fix GPU out of memory.
    4. Talk with Avi about the dataset - get a better one.
    5. Complete the other strategies.
    6. Test the model before training.
    7. Test the model after small amount of epochs training.
    8. Ask Amos if we want a lot of epochs - intentional overfiting?
    9. 
""" 
