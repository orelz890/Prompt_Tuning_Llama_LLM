from transformers import pipeline

# Load the instruct-tuned model
model_name = "unsloth/Llama-3.2-1B-Instruct"

# Initialize the pipeline
pipe = pipeline("text-generation", model=model_name)

# Main function for interactive chat
def main():
    print("[INFO] Entering instruct-style chat mode. Type 'exit' to quit.")

    while True:
        try:
            # Get user input (instruction)
            user_input = input("[USER]: ")
            if user_input.lower() in {"exit", "quit"}:
                print("[INFO] Exiting chat.")
                break

            # Format the input as an instruct-style message
            messages = [{"role": "user", "content": user_input}]

            # Generate response using the pipeline
            response = pipe(messages, max_length=100, do_sample=True, top_p=0.9, temperature=0.7)

            # Display the response
            print(f"[MODEL]: {response[0]['generated_text']}")

        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
            break


if __name__ == "__main__":
    main()
