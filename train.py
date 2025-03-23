from datasets import load_dataset, Dataset
import re
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score
import joblib

#Import dataset
all_product_dataset = load_dataset("json", data_files="amz_products_small.jsonl", split="train")
#Only select 10000 rows in the dataset to make the training faster
product_dataset = all_product_dataset.shuffle(seed=42).select(range(10000))

#Feature engineering
# Step 1: In order to make training faster we will remove all the columns except for description
columns_to_remove = ['title','feature','brand','also_buy', 'also_view', 'asin', 'category', 'image', 'price']
product_dataset = product_dataset.remove_columns(columns_to_remove)

# Step 2: Flatten lists in the dataset
def flatten_lists(example):
    for key, value in example.items():
        if isinstance(value, list):
            example[key] = ' '.join(value)
    return example

product_dataset = product_dataset.map(flatten_lists)

# Step 3: Lowercase text columns
def lowercase_text(example):
    text_columns = ['description']
    for col in text_columns:
        example[col] = example[col].lower()
    return example

product_dataset = product_dataset.map(lowercase_text)

# Step 4: Clean HTML tags from description
def clean_html(example):
    example['description'] = re.sub(r'<[^>]+>', '', example['description'])
    return example

# Step 5 Rename main_cat into label
product_dataset = product_dataset.rename_column(
    original_column_name="main_cat", new_column_name="label"
)

product_dataset = product_dataset.map(clean_html)

# Step 6: Load the tokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Step 7: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["description"],
        padding="max_length",  # Pad sequences to max_length
        truncation=True,       # Truncate sequences longer than max_length
        max_length=128,        # Set a fixed length for all sequences
    )

# Apply tokenization to the dataset
tokenized_datasets = product_dataset.map(tokenize_function, batched=True)

# Extract all main_cat values
main_cat_values = product_dataset['label']  # Adjust based on your dataset structure

# Fit the label encoder on all main_cat values
label_encoder = LabelEncoder()
label_encoder.fit(main_cat_values)

# Add encoded labels to the dataset
def encode_labels(example):
    return {'label': label_encoder.transform([example['label']])[0]}  # Transform single value

tokenized_datasets = tokenized_datasets.map(encode_labels)

#Save the label encoder for inference
joblib.dump(label_encoder, "label_encoder.pkl")

#Everything is ready we can train our model
# Step 1: Load the model
num_labels = len(label_encoder.classes_)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

# Step 2: Set up training argument
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
)

# Step 3: Define metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Step 4: Initialize the Trainer
tokenized_datasets = tokenized_datasets.train_test_split(train_size=0.8, seed=42)
# Initialize the data collator
data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Step 5: Train the model
trainer.train()

# Step 6: Evaluate the model
results = trainer.evaluate()
print(results)

# Step 7: Save the model
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")

