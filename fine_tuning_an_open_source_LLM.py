# -*- coding: utf-8 -*-
"""Fine tuning an open source LLM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xrYX-GKSTfMz4BwrAI08W7MjHlsRynOY
"""

!pip install peft

# Step 0: Install necessary libraries
!pip install transformers datasets accelerate bitsandbytes peft

# Import libraries
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.optim import AdamW  # Switch to PyTorch's AdamW optimizer

# Step 1: Load the medical text dataset
dataset = load_dataset('pubmed_qa', 'pqa_labeled', split='train')

# Step 2: Load GPT-2 model and tokenizer (can change to LLaMA if required)
model_name = "gpt2"  # If using LLaMA, replace this with your LLaMA model path
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the GPT-2 tokenizer has a padding token
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as PAD token

# Load GPT-2 model for Causal LM (language modeling)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 3: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['question'], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 4: Add labels for training
def add_labels(examples):
    examples["labels"] = examples["input_ids"].copy()  # GPT-2 uses input_ids as labels for language modeling
    return examples

tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)

# Step 5: Prepare the DataLoader
train_dataloader = DataLoader(tokenized_datasets, batch_size=8, shuffle=True)

# Step 6: Define the Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,  # For mixed precision training
)

# Step 7: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    optimizers=(AdamW(model.parameters(), lr=5e-5), None)  # Switch to standard PyTorch AdamW optimizer
)

# Step 8: Fine-tune the model
trainer.train()

# Step 9: Save the fine-tuned model
trainer.save_model("./fine-tuned-gpt2-medical")

# Step 10: Evaluate the model
results = trainer.evaluate()
print(f"Results: {results}")