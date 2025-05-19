#!/usr/bin/env python3
from measure_throughput import measure_throughput
import argparse
import json
import os
from datetime import datetime

def compare_models(base_model_id, full_model_id, prompt, max_new_tokens, num_runs):
    """
    Compare throughput between a full fine-tuned model and a LoRA model.
    
    Args:
        base_model_id: Base model ID for LoRA
        full_model_id: Full fine-tuned model ID
        prompt: Input prompt for generation
        max_new_tokens: Number of tokens to generate
        num_runs: Number of runs to average over
    """
    # Create results directory if it doesn't exist
    os.makedirs("throughput_results", exist_ok=True)
    
    # Generate timestamp for the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*50)
    print("COMPARING MODEL THROUGHPUT")
    print("="*50)
    
    print("\nTesting LoRA model...")
    lora_throughput = measure_throughput(
        base_model_id,
        prompt,
        max_new_tokens,
        num_runs,
        use_lora=True
    )
    
    print("\nTesting full fine-tuned model...")
    full_throughput = measure_throughput(
        full_model_id,
        prompt,
        max_new_tokens,
        num_runs,
        use_lora=False
    )
    
    # Calculate speed difference
    speedup = (lora_throughput / full_throughput - 1) * 100
    
    # Print comparison
    print("\n" + "="*50)
    print("THROUGHPUT COMPARISON RESULTS")
    print("="*50)
    print(f"LoRA model throughput:     {lora_throughput:.2f} tokens/second")
    print(f"Full model throughput:     {full_throughput:.2f} tokens/second")
    print(f"LoRA speedup:              {speedup:.2f}%")
    
    # Save results to a file
    results = {
        "timestamp": timestamp,
        "base_model_id": base_model_id,
        "full_model_id": full_model_id,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "num_runs": num_runs,
        "lora_throughput": lora_throughput,
        "full_throughput": full_throughput,
        "speedup_percentage": speedup
    }
    
    results_file = f"throughput_results/comparison_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare throughput between LoRA and full fine-tuned models")
    parser.add_argument("--base_model_id", type=str, required=True,
                        help="Base model ID for LoRA (e.g., 'meta-llama/Llama-3.2-1B-Instruct-hf')")
    parser.add_argument("--full_model_id", type=str, required=True,
                        help="Full fine-tuned model ID (e.g., 'Pongsaky/llama3.2-typhoon2-1b-full-training-phonetic')")
    parser.add_argument("--prompt", type=str, default="เขียนกลอนแปดเกี่ยวกับความรัก",
                        help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Number of tokens to generate")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of runs to average over")
    
    args = parser.parse_args()
    
    compare_models(
        args.base_model_id,
        args.full_model_id,
        args.prompt,
        args.max_new_tokens,
        args.num_runs
    ) 