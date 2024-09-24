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
- [License](#license)

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

```bash
pip install transformers datasets accelerate bitsandbytes peft

#Make sure you also have torch installed, either with CPU or GPU support:
pip install torch

## Dataset
The dataset used is the PubMed QA dataset, specifically the pqa_labeled version. This dataset contains question-answer pairs in the medical domain, which makes it ideal for fine-tuning GPT-2 for domain-specific question answering.

Load Datasetfrom datasets import load_dataset
dataset = load_dataset('pubmed_qa', 'pqa_labeled', split='train')


## Results
After fine-tuning, the model can generate medical domain-specific text and answer questions based on the PubMed QA dataset. The performance of the model is evaluated using standard metrics, and results are printed.

Usage
Once the model is trained, you can use it for inference on new medical text questions:
input_text = "What are the symptoms of COVID-19?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

