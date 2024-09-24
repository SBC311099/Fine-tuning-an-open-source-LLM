# Fine-Tuning GPT-2 for Medical Text Processing

This project demonstrates how to fine-tune the GPT-2 model on the **PubMed QA** dataset, specifically the `pqa_labeled` version, for medical text generation and question-answering tasks.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)

## Introduction
In this project, we fine-tune GPT-2 for question answering on a medical domain dataset called **PubMed QA**. We use Hugging Face's `transformers` library to adapt GPT-2, a language model pre-trained on general data, for specialized tasks in the medical field. This fine-tuning improves the model's performance in generating relevant and coherent answers to medical questions.

## Requirements
- Python 3.8+
- Libraries:
  - `transformers`
  - `datasets`
  - `accelerate`
  - `bitsandbytes`
  - `peft` (optional for parameter-efficient fine-tuning)
  - `torch` (for model training)
  - `tqdm` (for progress tracking)

## Installation

To run this project, first install the necessary libraries by running:

```python pip install transformers datasets accelerate bitsandbytes peft torch```



## Dataset
The dataset used is the PubMed QA dataset, specifically the pqa_labeled version. This dataset contains question-answer pairs in the medical domain, which makes it ideal for fine-tuning GPT-2 for domain-specific question answering.

python
Copy code
from datasets import load_dataset

# Load PubMed QA dataset
dataset = load_dataset('pubmed_qa', 'pqa_labeled', split='train')
Model Training
Step 1: Load GPT-2 Model and Tokenizer
We use the pre-trained GPT-2 model and tokenizer from Hugging Face's transformers library.

python
Copy code
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load GPT-2 tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the GPT-2 tokenizer has a padding token
tokenizer.pad_token = tokenizer.eos_token
Step 2: Preprocess the Data
We tokenize the input data (the questions from the PubMed QA dataset) and pad or truncate the sequences to a maximum length of 128 tokens.

python
Copy code
# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['question'], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
Step 3: Add Labels for Training
We add labels to the tokenized dataset for language modeling tasks.

python
Copy code
# Add labels for the language modeling task
def add_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)
Step 4: Prepare the DataLoader
We prepare the DataLoader for training, using a batch size of 8.

python
Copy code
from torch.utils.data import DataLoader

# Prepare data loader
train_dataloader = DataLoader(tokenized_datasets, batch_size=8, shuffle=True)
Step 5: Define the Training Arguments
We set up the training arguments, including evaluation strategy, learning rate, and the number of training epochs.

python
Copy code
from transformers import TrainingArguments

# Set up training arguments
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
    fp16=True  # For mixed precision training
)
Step 6: Initialize the Trainer
We initialize the Trainer class from the Hugging Face transformers library, which handles the training loop and optimization.

python
Copy code
from transformers import Trainer
from torch.optim import AdamW

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    optimizers=(AdamW(model.parameters(), lr=5e-5), None)
)
Step 7: Fine-Tune the Model
We begin fine-tuning the GPT-2 model on the medical question-answering dataset.

python
Copy code
# Fine-tune the model
trainer.train()
Step 8: Save the Fine-Tuned Model
After training, we save the fine-tuned model for future use.

python
Copy code
# Save the fine-tuned model
trainer.save_model("./fine-tuned-gpt2-medical")
Evaluation
Once the model is trained, you can evaluate it on the validation dataset:

python
Copy code
# Evaluate the model
results = trainer.evaluate()
print(f"Results: {results}")
Results
After fine-tuning, the model is capable of generating medical domain-specific text and answering questions based on the PubMed QA dataset. The performance of the model can be evaluated using various metrics such as loss and accuracy, which are printed out after the evaluation.

Usage
Once the model is trained, you can use it for inference on new medical text questions:

python
Copy code
input_text = "What are the symptoms of COVID-19?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
arduino
Copy code

In this format, after each **code block**, the section headings and text will render as **bol
