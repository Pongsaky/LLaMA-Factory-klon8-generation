from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import argparse
import os

def measure_throughput(model_id, prompt, max_new_tokens=100, num_runs=5, use_lora=False, lora_weights=None):
    """
    Measure the throughput (tokens/second) of a model during text generation.
    
    Args:
        model_id: HuggingFace model ID or local path
        prompt: Input text to generate from
        max_new_tokens: Number of tokens to generate
        num_runs: Number of runs to average over
        use_lora: Whether to use LoRA weights
        lora_weights: Path to LoRA weights (if use_lora=True)
    
    Returns:
        avg_tokens_per_second: Average tokens per second
    """
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Configure model loading based on whether LoRA is used
    if use_lora:
        from peft import AutoPeftModelForCausalLM
        
        print(f"Loading base model for LoRA")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            tie_word_embeddings=False,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = model.merge_and_unload()
        
    else:
        print("Loading full model")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Ensure model is in evaluation mode
    model.eval()
    
    system_prompt = "You are an expert Thai poet specializing in `กลอนแปด` (Eight-syllable verse) poetry. When a user provides a prompt, respond ONLY with a Thai poem (2-4 stanzas) that addresses their request, without any explanations or commentary. Your poem must strictly follow traditional กลอนแปด structure and rhyming patterns. Include phonetic rhyming tags for all rhyming words using the format: `<r>[vowel][ending consonant]word</r>` (examples: `<r>[a][w]เขา</r>`, `<r>[o][k]นก</r>`, `<r>[a]ผา</r>`, `<r>[i]ศรี</r>`). These tags should mark all external and internal rhymes according to proper กลอนแปด structure. Create vivid, culturally appropriate poetry that demonstrates mastery of Thai prosody while faithfully addressing the user's requested theme or scenario."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Tokenize the prompt
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,padding=True, return_tensors="pt").to(model.device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(model.device)
    # Warm-up run to ensure GPU is at full speed
    print("Performing warm-up run...")
    with torch.no_grad():
        _ = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7
        )
    
    # Perform timed runs
    tokens_per_second_list = []
    
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}...")
        
        # Start timer
        start_time = time.time()
        
        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )
        
        # End timer
        end_time = time.time()
        
        # Calculate number of new tokens generated
        num_new_tokens = output.shape[1] - input_ids.shape[1]
        
        # Calculate time taken
        time_taken = end_time - start_time
        
        # Calculate tokens per second
        tokens_per_second = num_new_tokens / time_taken
        tokens_per_second_list.append(tokens_per_second)
        
        print(f"  Generated {num_new_tokens} tokens in {time_taken:.2f} seconds")
        print(f"  Throughput: {tokens_per_second:.2f} tokens/second")
        
        # Print a sample of the generated text for the first run
        if run == 0:
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"\nSample generation:\n{generated_text}\n")
    
    # Calculate average tokens per second
    avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list)
    
    print(f"\nAverage throughput: {avg_tokens_per_second:.2f} tokens/second")
    return avg_tokens_per_second

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure model throughput in tokens/second")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--prompt", type=str, default="เขียนกลอนแปดเกี่ยวกับความรัก", help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs to average over")
    parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA weights")
    parser.add_argument("--lora_weights", type=str, help="Path to LoRA weights (required if use_lora=True)")
    
    args = parser.parse_args()
    
    if args.use_lora and args.lora_weights is None:
        parser.error("--lora_weights is required when --use_lora is set")
    
    measure_throughput(
        args.model_id,
        args.prompt,
        args.max_new_tokens,
        args.num_runs,
        args.use_lora,
        args.lora_weights
    ) 