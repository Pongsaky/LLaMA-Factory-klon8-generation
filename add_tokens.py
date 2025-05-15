#!/usr/bin/env python
import argparse
import os
import json
import torch
import gc
import re
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_data_from_json(file_path):
    """Load data from a JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def extract_phonetic_combinations(training_data, tokenizer, model_type="llama"):
    """Extract phonetic combinations from training data"""
    assert(model_type in ["gemma", "llama"])
    # Define regex pattern to match <r>[X][Y]text</r> patterns
    pattern = r'<r>(.+?)</r>'

    tag_dict = defaultdict(set)

    # Process each text in the training data
    for data in training_data:
        # Find all matches of the pattern in the text
        text = data["text"]
        matches = re.findall(pattern, text)

        for item in matches:
            # Extract all tags (e.g., [a], [w])
            tags = re.findall(r'\[.*?\]', item)
            # Extract the word by removing tags
            word = re.sub(r'\[.*?\]', '', item).strip()
            # Add the word to each tag's set
            for tag in tags:
                if "<r>" in word:
                    print(f"Warning: <r> tag found in word '{word}'. Skipping this entry.")
                    continue
                tag_dict[tag].add(word)

    new_tag_dict = defaultdict(set)
    for key, value in tag_dict.items():
        for word in value:
            if model_type == "gemma":
                tokenized_words = tokenizer.tokenizer.encode(word, add_special_tokens=False)
            else:
                tokenized_words = tokenizer.encode(word, add_special_tokens=False)
            
            if "<r>" in tokenized_words:
                print(f"Value: {value}")
                print(f"Tokenized word: {word}")
                print(f"Tokenized word: {tokenized_words}")
            
            for tokenized_word in tokenized_words:
                new_tag_dict[key].add(tokenized_word)

    return new_tag_dict

def mean_surround_new_tag_token(model, tag_dict):
    """Calculate the mean of surrounding token embeddings"""
    embedding_matrix = model.get_input_embeddings().weight.clone()
    lm_head_matrix = model.get_output_embeddings().weight.clone()

    tag_embedding_dict = {}
    tag_lm_head_dict = {}
    
    tag_embedding = torch.zeros_like(embedding_matrix[0])
    tag_lm_head = torch.zeros_like(lm_head_matrix[0])

    for tag, ids_values in tag_dict.items():
        # properly accumulate all the embeddings
        tag_embedding.zero_()
        tag_lm_head.zero_()
        for idx in ids_values:
            tag_embedding += embedding_matrix[idx]
            tag_lm_head += lm_head_matrix[idx]
        tag_embedding_dict[tag] = tag_embedding.clone() / len(ids_values)
        tag_lm_head_dict[tag] = tag_lm_head.clone() / len(ids_values)

    return tag_embedding_dict, tag_lm_head_dict

def add_new_tokens_smart(model, tokenizer, tag_dict, new_tokens, model_type="llama"):
    """
    Smartly resizes the tokenizer and adds new tokens to the model using mean of related tokens.
    """
    assert isinstance(new_tokens, (list, tuple))
    assert len(new_tokens) > 0
    assert len(tag_dict) > 0
    assert isinstance(tag_dict, dict)
    assert model_type in ["gemma", "llama"]

    # Check if tokens already exist
    if model_type == "gemma":
        overlapping_tokens = set(new_tokens) & set(tokenizer.tokenizer.vocab.keys())
    else:
        overlapping_tokens = set(new_tokens) & set(tokenizer.vocab.keys())
        
    if len(overlapping_tokens) != 0:
        print(
            f"You're adding new_tokens = {new_tokens}\n"
            f"There are tokens which are overlapping = {list(overlapping_tokens)}\n"
            f"We shall safely ignore these overlapping tokens."
        )
        new_tokens = [x for x in new_tokens if x not in overlapping_tokens]

    # Calculate mean embeddings
    tag_embedding_dict, tag_lm_head_dict = mean_surround_new_tag_token(model, tag_dict)

    # Get old lengths
    old_input_embedding = model.get_input_embeddings().weight
    old_output_embedding = model.get_output_embeddings().weight
    old_input_length = old_input_embedding.shape[0]
    old_output_length = old_output_embedding.shape[0]
    if model_type == "gemma":
        old_config_size = model.config.text_config.vocab_size
    else:
        old_config_size = model.config.vocab_size

    # Check for tied weights
    is_tied = (old_input_embedding.data_ptr() == old_output_embedding.data_ptr()) \
        or (model.config.tie_word_embeddings)

    # Add tokens!
    if model_type == "gemma":
        old_length = len(tokenizer.tokenizer)
        tokenizer.tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer.tokenizer))
    else:
        old_length = len(tokenizer)
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))

    # Get the updated embedding matrices
    embedding_matrix = model.get_input_embeddings().weight
    lm_head_matrix = model.get_output_embeddings().weight

    # Confirm sizes are correct
    if embedding_matrix.shape[0] > (old_input_length + len(new_tokens)):
        raise RuntimeError("Embedding matrix size did not get resized properly!")
    if lm_head_matrix.shape[0] > (old_output_length + len(new_tokens)):
        raise RuntimeError("LM Head matrix size did not get resized properly!")
    if model_type == "gemma":
        if model.config.text_config.vocab_size > (old_config_size + len(new_tokens)):
            raise RuntimeError("Model's config vocab_size did not get resized properly!")
    else:
        if model.config.vocab_size > (old_config_size + len(new_tokens)):
            raise RuntimeError("Model's config vocab_size did not get resized properly!")

    # Set embeddings for new tokens
    key_list = list(tag_dict.keys())
    if model_type == "gemma":
        key_ids_list = [tokenizer.tokenizer.encode(word, add_special_tokens=False)[0] for word in key_list]
    else:
        key_ids_list = [tokenizer.encode(word, add_special_tokens=False)[0] for word in key_list]
    
    with torch.no_grad():
        for key, ids in zip(key_list, key_ids_list):
            tag_embedding = tag_embedding_dict[key]
            tag_lm_head = tag_lm_head_dict[key]
            embedding_matrix[ids] = tag_embedding
            lm_head_matrix[ids] = tag_lm_head

    # Mark embeddings for training
    internal_model = model
    while hasattr(internal_model, "model"):
        internal_model._need_to_train_embeddings = True
        internal_model = internal_model.model
    internal_model._need_to_train_embeddings = True
    
    # Update vocab sizes in config
    current_model = model
    if hasattr(current_model, "model") and hasattr(current_model, "config"):
        if model_type == "gemma":
            if hasattr(current_model.config.text_config, "vocab_size"):
                current_model.config.text_config.update({"vocab_size": len(tokenizer.tokenizer)})
        else:
            if hasattr(current_model.config, "vocab_size"):
                current_model.config.update({"vocab_size": len(tokenizer)})
        current_model = current_model.model

    # Re-tie weights if they were tied
    if is_tied:
        model.tie_weights()

    # Clear GPU memory
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()

def add_new_tokens_standard(model, tokenizer, new_tokens, model_type="llama"):
    """
    Standard approach to add new tokens using tokenizer.add_tokens and model.resize_token_embeddings
    """
    print(f"Adding {len(new_tokens)} new tokens using standard approach...")
    
    # Check if tokens already exist
    if model_type == "gemma":
        overlapping_tokens = set(new_tokens) & set(tokenizer.tokenizer.vocab.keys())
        if len(overlapping_tokens) != 0:
            print(f"Ignoring {len(overlapping_tokens)} overlapping tokens")
            new_tokens = [x for x in new_tokens if x not in overlapping_tokens]
            
        # Add tokens and resize
        tokenizer.tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer.tokenizer))
    else:
        overlapping_tokens = set(new_tokens) & set(tokenizer.vocab.keys())
        if len(overlapping_tokens) != 0:
            print(f"Ignoring {len(overlapping_tokens)} overlapping tokens")
            new_tokens = [x for x in new_tokens if x not in overlapping_tokens]
            
        # Add tokens and resize
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
        
    print("Finished adding new tokens.")

def save_model(model, tokenizer, output_path, push_to_hub=False, hub_model_id=None):
    """Save the model and tokenizer locally or to HuggingFace Hub"""
    if push_to_hub:
        if hub_model_id is None:
            raise ValueError("hub_model_id must be provided when push_to_hub is True")
        print(f"Pushing model to HuggingFace Hub as {hub_model_id}...")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        print(f"Model and tokenizer pushed to {hub_model_id}")
    else:
        print(f"Saving model to {output_path}...")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Model and tokenizer saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Add new tokens to a language model")
    
    parser.add_argument("--model_name", type=str, required=True,
                        help="Path or name of the pre-trained model")
    
    parser.add_argument("--model_type", type=str, choices=["llama", "gemma"], default="llama",
                        help="Type of model (llama or gemma)")
    
    parser.add_argument("--method", type=str, choices=["standard", "smart"], required=True,
                        help="Method to use for adding tokens: standard or smart")
    
    parser.add_argument("--training_data", type=str, required=True,
                        help="Path to training data JSON file")
    
    parser.add_argument("--new_tokens", type=str, required=True,
                        help="Path to new tokens JSON file")
    
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the model and tokenizer")
    
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to HuggingFace Hub instead of saving locally")
    
    parser.add_argument("--hub_model_id", type=str,
                        help="Model ID for HuggingFace Hub (required if push_to_hub is True)")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load data
    print(f"Loading training data from {args.training_data}...")
    training_data = get_data_from_json(args.training_data)
    
    print(f"Loading new tokens from {args.new_tokens}...")
    new_tokens = get_data_from_json(args.new_tokens)
    
    # Add new tokens
    if args.method == "standard":
        add_new_tokens_standard(model, tokenizer, new_tokens, args.model_type)
    elif args.method == "smart":
        # Extract phonetic combinations first
        print("Extracting phonetic combinations from training data...")
        tag_dict = extract_phonetic_combinations(training_data, tokenizer, args.model_type)
        print(f"Found {len(tag_dict)} phonetic tags.")
        
        # Add tokens using smart initialization
        print(f"Adding {len(new_tokens)} new tokens using smart initialization...")
        add_new_tokens_smart(model, tokenizer, tag_dict, new_tokens, args.model_type)
        print("Finished adding new tokens.")
    
    # Save model and tokenizer
    save_model(model, tokenizer, args.output_path, args.push_to_hub, args.hub_model_id)

if __name__ == "__main__":
    main() 