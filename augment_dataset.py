from datasets import load_dataset, Dataset, concatenate_datasets

# Load dataset
dataset = load_dataset("OrelZamler/Human_Conversations")

# Extract train set
train_data = dataset['train']

# Storage for new rows
new_rows = []

# Variables to track the conversation
room = None
last_message = None 

for idx, data in enumerate(train_data):
    
    if room == data['PairId']:
        new_rows.append({
            'PairId': data['PairId'], 
            'Questions': last_message, 
            'Answers': data['Questions']
        })

    else:
        room = data['PairId']
        
    last_message = data['Answers']

# Convert new_rows list to Dataset format only if there are new rows
if new_rows:
    new_dataset = Dataset.from_list(new_rows)
    # Concatenate the new dataset with the original dataset
    updated_dataset = concatenate_datasets([train_data, new_dataset])
else:
    updated_dataset = train_data  # No new rows, keep the dataset unchanged

# Print updated dataset
print(updated_dataset)

for x in updated_dataset:
    print(x['PairId'], x['Questions'], x['Answers'])


# Save dataset to CSV
csv_filename = "updated_dataset.csv"
updated_dataset.to_csv(csv_filename)

print(f"Dataset saved as {csv_filename}")