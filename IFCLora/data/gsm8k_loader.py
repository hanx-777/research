from datasets import load_dataset
from transformers import PreTrainedTokenizer
import torch

def get_gsm8k_dataset(tokenizer: PreTrainedTokenizer):
    """
    Loads and tokenizes the GSM8K dataset for causal language modeling.
    Masks the prompt during training.
    """
    def preprocess_fn(ex):
        prompt = f"Question: {ex['question']}\nAnswer:"
        answer = f" {ex['answer']} <|endoftext|>"
        
        full_text = prompt + answer
        tokenized = tokenizer(full_text, max_length=512, truncation=True, padding=False)
        
        # Labels: -100 for the prompt tokens, actual tokens for the answer
        prompt_ids = tokenizer(prompt, max_length=512, truncation=True)["input_ids"]
        labels = [-100] * len(prompt_ids) + tokenized["input_ids"][len(prompt_ids):]
        
        return {
            "input_ids": tokenized["input_ids"],
            "labels": labels
        }

    ds = load_dataset("gsm8k", "main")
    tokenized_ds = ds.map(
        preprocess_fn, 
        remove_columns=ds["train"].column_names,
        desc="Tokenizing GSM8K"
    )
    return tokenized_ds
