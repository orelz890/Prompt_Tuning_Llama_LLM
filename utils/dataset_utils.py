from datasets import load_dataset

def preprocess_and_tokenize(dataset_path, tokenizer):
    # Load dataset
    dataset = load_dataset(dataset_path)

    # Split into train (80%), eval (16%), and test (4%)
    split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
    test_dataset = split_dataset["test"]

    train_test_split = split_dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    def tokenize_function(examples):
        # Convert questions and answers to strings
        questions = [str(q) if q is not None else "" for q in examples["Questions"]]
        answers = [str(a) if a is not None else "" for a in examples["Answers"]]

        # Tokenize inputs (questions)
        inputs = tokenizer(
            questions,
            truncation=True,
            padding=False,  # No padding here
            max_length=512,
        )

        # Tokenize targets (answers) as labels
        labels = tokenizer(
            text_target=answers,
            truncation=True,
            padding=False,  # No padding here
            max_length=512,
        )["input_ids"]

        # TODO - check this padding
        # Ensure `labels` match `input_ids` length
        for i in range(len(labels)):
            input_len = len(inputs["input_ids"][i])
            label_len = len(labels[i])

            if label_len < input_len:
                # Pad labels to match input length
                labels[i] += [-100] * (input_len - label_len)
            elif label_len > input_len:
                # Truncate labels to match input length
                labels[i] = labels[i][:input_len]

        # Add labels to inputs
        inputs["labels"] = labels
        return inputs

    # Tokenize the datasets using map()
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    sample = tokenized_train_dataset[0]
    print("Input IDs:", sample["input_ids"])
    print("Labels:", sample["labels"])
    print("Attention Mask:", sample["attention_mask"])

    # Display example
    # print("tokenized_train_dataset example[:5]: ", tokenized_train_dataset[:5])
    
    return tokenized_train_dataset, tokenized_eval_dataset, tokenized_test_dataset
