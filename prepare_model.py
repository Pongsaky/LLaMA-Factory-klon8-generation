import os
import yaml
import torch
import argparse
import numpy as np
import psutil
import shutil # For file backup
from pathlib import Path

# (get_gpu_memory_info, get_ram_memory_info, calculate_model_memory, 
#  calculate_dataset_memory, estimate_model_size_from_params functions remain the same as your last provided version)
# For brevity, I'll assume these functions are defined as in your previous version.
# Make sure to include them when you run the script.

def get_gpu_memory_info():
    """Returns detailed GPU memory information in GB"""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU mode")
        return {
            "total": 0,
            "reserved": 0,
            "allocated": 0,
            "available": 0,
            "raw_available": 0
        }
    
    gpu_id = 0 
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
    # More accurate available memory: total - reserved (what CUDA context + others hold)
    # memory_allocated is what the current Pytorch tensors hold.
    # memory_reserved includes memory_allocated + cached memory by Pytorch allocator
    reserved_cuda_memory = torch.cuda.memory_reserved(gpu_id) / (1024**3)
    allocated_cuda_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
    
    # The memory truly available for new allocations is total_memory - reserved_cuda_memory
    available_for_new_allocations = total_memory - reserved_cuda_memory
    
    return {
        "total": total_memory,
        "reserved_pytorch": reserved_cuda_memory, # Memory reserved by PyTorch's caching allocator
        "allocated_pytorch": allocated_cuda_memory, # Memory currently used by PyTorch tensors
        "available_for_new_pytorch_alloc": available_for_new_allocations, # Total - PyTorch reserved
        "raw_available": torch.cuda.mem_get_info()[0] / (1024**3) # Free memory reported by nvidia-smi
    }

def get_ram_memory_info():
    """Returns system RAM memory information in GB"""
    vm = psutil.virtual_memory()
    return {
        "total": vm.total / (1024**3),
        "available": vm.available / (1024**3),
        "used": vm.used / (1024**3),
        "percent": vm.percent
    }

def calculate_model_memory(model_size_gb_fp32, precision="float16", use_lora=False, use_qlora=False, quantization=None, lora_rank=8, model_hidden_size=4096):
    """
    Calculate detailed model memory requirements in GB
    model_size_gb_fp32: Model size as if it were in FP32 (e.g., 1B params * 4 bytes/param)
    """
    bytes_per_param_fp32 = 4
    
    # Calculate base model weights memory
    if use_qlora:
        if quantization == '4bit':
            model_weights_memory = model_size_gb_fp32 * 0.125 
        elif quantization == '8bit':
            model_weights_memory = model_size_gb_fp32 * 0.25
        else: # Should not happen if validation is correct
            model_weights_memory = model_size_gb_fp32 * 0.5 # Default to fp16 if QLoRA but no quant
    elif precision == "float32":
        model_weights_memory = model_size_gb_fp32 
    elif precision in ["float16", "bfloat16"]:
        model_weights_memory = model_size_gb_fp32 * 0.5
    else: # Should not happen
        model_weights_memory = model_size_gb_fp32 * 0.5

    # Optimizer states memory (AdamW typically needs 2 states per parameter)
    # Each state is usually the same precision as training, or FP32 for some optimizers.
    # For simplicity, assume optimizer states match training precision for trainable params.
    bytes_per_optimizer_state_param = 2 if precision in ["float16", "bfloat16"] else 4
    
    if use_lora or use_qlora:
        # Estimate LoRA params: 2 * rank * (input_dim + output_dim).
        # Assuming input_dim ~ output_dim ~ model_hidden_size. For multiple layers, sum this up.
        # A very rough estimate: LoRA params are a small fraction (e.g. 0.1-1%) of total params.
        # Let's say LoRA trainable params are ~ (lora_rank / model_hidden_size) * total_params, but this is per layer.
        # A simpler heuristic: total LoRA params often ~20-100M for 7B-70B models.
        # For a 7B model (fp32 size ~28GB), 28M LoRA params in fp16 = 56MB.
        # Let's assume LoRA parameters are 0.5% of the full model's parameters for estimation.
        # And for those, we need optimizer states.
        num_full_model_params = (model_size_gb_fp32 * (1024**3)) / bytes_per_param_fp32
        num_lora_params_approx = num_full_model_params * (lora_rank / 512) * 0.1 # Very rough heuristic
        optimizer_memory = (num_lora_params_approx * bytes_per_optimizer_state_param * 2) / (1024**3) # 2 states for Adam
    else: # Full fine-tuning
        optimizer_memory = model_weights_memory * 2 # 2 states (m, v) for Adam-like optimizers

    # Activation memory (highly dependent on seq_length, batch_size, hidden_size, num_layers, and grad checkpointing)
    # Without gradient checkpointing, activations can be roughly: batch_size * seq_length * num_layers * hidden_size * bytes_per_activation
    # With gradient checkpointing, it's much lower: sqrt(num_layers) factor.
    # For simplicity, let's use a fraction of model_weights_memory, higher if no grad checkpointing.
    # This is a very rough part.
    activation_memory_factor = 0.2 if True else 1.0 # Assuming gradient_checkpointing is True for now
    activation_memory = model_weights_memory * activation_memory_factor
    
    # Forward pass temporary memory (e.g. intermediate results, not KV cache yet)
    forward_temp_memory = model_weights_memory * 0.1 # Small fraction
    
    # Summing up model-related memory (excluding per-batch data like KV cache for full batch)
    # KV cache will be handled in per-batch calculation.
    total_static_model_memory = model_weights_memory + optimizer_memory + activation_memory + forward_temp_memory
    
    overhead_factor = 1.1 # 10% general overhead
    total_static_model_memory *= overhead_factor
    
    return {
        "model_weights": model_weights_memory,
        "optimizer_states": optimizer_memory,
        "activations_approx": activation_memory, # Emphasize approximation
        "forward_temp": forward_temp_memory,
        "total_static_model_memory": total_static_model_memory
    }

def calculate_dataset_memory(dataset_size, seq_length, batch_size, dataloader_workers=2):
    """
    Calculate memory required for dataset in GB (mainly for CPU RAM impact)
    """
    bytes_per_token = 4 # Typically token IDs are int32 or int64
    bytes_per_example = seq_length * bytes_per_token * 2 # input_ids, attention_mask
    
    # For dataloader prefetching in RAM
    prefetch_factor = dataloader_workers + 1 
    prefetched_examples = batch_size * prefetch_factor
    prefetch_memory_ram_gb = (prefetched_examples * bytes_per_example) / (1024**3)
    
    # For potentially memory-mapped or cached parts of the dataset in RAM
    # This is highly dependent on how the dataset is loaded and if it's small enough to fit.
    # Assume a small constant RAM overhead for dataset handling itself.
    dataset_handling_ram_gb = 0.5 # Heuristic for dataset object, tokenizers etc. in RAM
    
    total_dataset_ram_impact_gb = prefetch_memory_ram_gb + dataset_handling_ram_gb
    
    # GPU memory for one batch of data (input_ids, attention_mask, labels)
    # This will be transferred to GPU.
    gpu_batch_data_gb = (batch_size * bytes_per_example) / (1024**3)

    return {
        "ram_impact_gb": total_dataset_ram_impact_gb,
        "gpu_batch_data_gb": gpu_batch_data_gb # Per batch, before model processing
    }


def calculate_batch_memory_per_sample(seq_length, model_hidden_size, num_model_layers, precision="float16", gradient_checkpointing=True):
    """
    Estimate GPU memory per sample in a batch during model forward/backward.
    Includes activations (if not fully checkpointed) and KV cache.
    """
    bytes_per_element = 2 if precision in ["float16", "bfloat16"] else 4

    # KV Cache: 2 (key, value) * num_layers * seq_length * model_hidden_size * bytes_per_element
    kv_cache_per_sample = 2 * num_model_layers * seq_length * model_hidden_size * bytes_per_element
    
    # Activations (very rough, especially with gradient checkpointing)
    # If gradient_checkpointing, it's roughly O(sqrt(num_layers) * seq_len * hidden_size)
    # Otherwise O(num_layers * seq_len * hidden_size)
    # Let's use a heuristic based on hidden_size and seq_length
    activation_footprint_factor = 0.5 if gradient_checkpointing else 2.0 
    # Number of FWD activations per token is roughly proportional to hidden_dim
    activations_per_sample = activation_footprint_factor * num_model_layers * seq_length * model_hidden_size * bytes_per_element * 0.1 # Heuristic scaling

    # Gradients for activations (if not recomputed)
    gradients_per_sample = activations_per_sample # Similar size to activations

    # Total per sample in batch (GPU)
    total_per_sample_gb = (kv_cache_per_sample + activations_per_sample + gradients_per_sample) / (1024**3)
    
    return total_per_sample_gb

def calculate_batch_settings(
    model_size_gb_fp32, available_gpu_memory_gb, 
    seq_length=2048, model_hidden_size=4096, num_model_layers=32, # Added model arch params
    use_lora=False, use_qlora=False, quantization=None, lora_rank=8,
    precision="float16", dataset_size=1000, 
    dataloader_workers=2, system_reserve_gpu_gb=1.0, # Reduced default system reserve
    gradient_checkpointing=True
):
    """Calculate appropriate batch size and gradient accumulation steps with detailed memory accounting"""
    
    model_memory_info = calculate_model_memory(
        model_size_gb_fp32, 
        precision=precision,
        use_lora=use_lora, 
        use_qlora=use_qlora, 
        quantization=quantization,
        lora_rank=lora_rank,
        model_hidden_size=model_hidden_size
    )
    
    # Memory for one batch of raw data (input_ids, labels, etc.) on GPU
    # Assuming tokens are int32/int64 (4-8 bytes), and we have input_ids, attention_mask, labels
    bytes_per_token_raw_data = 4 
    gpu_data_per_sample_gb = (seq_length * bytes_per_token_raw_data * 3) / (1024**3)

    # Memory per sample due to model processing (KV cache, activations not covered by static model memory)
    model_processing_per_sample_gb = calculate_batch_memory_per_sample(
        seq_length, model_hidden_size, num_model_layers, precision, gradient_checkpointing
    )
    
    total_memory_per_sample_in_batch_gb = gpu_data_per_sample_gb + model_processing_per_sample_gb

    # Calculate remaining memory for dynamic batch processing part
    memory_for_batches_gb = available_gpu_memory_gb - model_memory_info["total_static_model_memory"] - system_reserve_gpu_gb
    
    print(f"\nMemory Breakdown (GB):")
    print(f"  Available GPU for PyTorch: {available_gpu_memory_gb:.2f}")
    print(f"  Static Model Memory (weights, optimizer, static activations): {model_memory_info['total_static_model_memory']:.2f}")
    print(f"    - Model Weights: {model_memory_info['model_weights']:.2f}")
    print(f"    - Optimizer States: {model_memory_info['optimizer_states']:.2f}")
    print(f"  System Reserve on GPU: {system_reserve_gpu_gb:.2f}")
    print(f"  Memory Remaining for Batches: {memory_for_batches_gb:.2f}")
    print(f"  Estimated GPU memory per sample in batch (data + KV + dynamic activations): {total_memory_per_sample_in_batch_gb:.4f}")

    if memory_for_batches_gb <= 0 or total_memory_per_sample_in_batch_gb <= 1e-9: # Check for non-positive or zero
        print(f"\nWarning: Not enough memory for training with current settings, or per-sample memory is zero/negative!")
        print(f"  Static model memory ({model_memory_info['total_static_model_memory']:.2f} GB) + system reserve ({system_reserve_gpu_gb:.2f} GB) exceeds available GPU memory ({available_gpu_memory_gb:.2f} GB).")
        print(f"Suggestions to reduce memory:")
        print(f"  - Enable QLoRA with 4-bit quantization (--qlora --quantization 4bit).")
        print(f"  - Enable gradient checkpointing (--gradient_checkpointing).")
        print(f"  - Reduce sequence length (--seq_length). Current: {seq_length}")
        print(f"  - Reduce LoRA rank (--lora_rank) if using LoRA/QLoRA. Current: {lora_rank}")
        print(f"  - Increase system reserve if external processes are using GPU memory heavily.")
        print(f"  - Ensure model parameters (size, hidden_size, num_layers) are correctly specified for your model.")
        return 1, 64  # Fallback: Minimum batch size, high accumulation

    max_batch_size = max(1, int(memory_for_batches_gb / total_memory_per_sample_in_batch_gb))
    
    # Prefer power of 2 batch sizes, but not strictly necessary. Let's aim for something reasonable.
    if max_batch_size > 128: batch_size = 128
    elif max_batch_size > 64: batch_size = 64
    elif max_batch_size > 32: batch_size = 32
    elif max_batch_size > 16: batch_size = 16
    elif max_batch_size > 8: batch_size = 8
    elif max_batch_size > 4: batch_size = 4
    elif max_batch_size > 2: batch_size = 2
    else: batch_size = 1
    
    batch_size = max(1, min(batch_size, max_batch_size)) # Ensure it's within feasible range

    # Target effective batch size
    target_effective_batch = 32 
    if use_qlora or use_lora: target_effective_batch = 64 # Can often use larger effective batch with PEFT
    if available_gpu_memory_gb < 12: target_effective_batch = max(16, target_effective_batch // 2)
    if available_gpu_memory_gb < 8: target_effective_batch = max(8, target_effective_batch //2)


    accumulation_steps = max(1, int(np.ceil(target_effective_batch / batch_size)))
    
    print(f"\nBatch Settings Calculation:")
    print(f"  Max possible batch size (fitting dynamic parts): {max_batch_size}")
    print(f"  Selected batch size (heuristic): {batch_size}")
    print(f"  Target effective batch size: {target_effective_batch}")
    print(f"  Calculated gradient accumulation steps: {accumulation_steps}")
    final_effective_batch_size = batch_size * accumulation_steps
    print(f"  Final effective batch size: {final_effective_batch_size}")

    if final_effective_batch_size < 4 :
        print("Warning: Effective batch size is very small. Training might be unstable.")

    return batch_size, accumulation_steps


def calculate_training_steps(total_samples, batch_size, accumulation_steps, epochs):
    """Calculate training steps and related parameters"""
    if batch_size == 0 or accumulation_steps == 0:
        print("Error: Batch size or accumulation steps is zero, cannot calculate training steps.")
        return { # Return default/error values
            "total_steps": 0, "steps_per_epoch": 0, "warmup_steps": 0,
            "save_steps": 0, "eval_steps": 0, "logging_steps": 0
        }
        
    steps_per_epoch = total_samples // (batch_size * accumulation_steps)
    if steps_per_epoch == 0:
        print("Warning: Calculated steps_per_epoch is 0. Effective batch size may be larger than dataset size for one epoch.")
        print("Setting steps_per_epoch to 1. Consider reducing epochs or increasing dataset size.")
        steps_per_epoch = 1 
        
    total_steps = steps_per_epoch * epochs
    warmup_ratio = 0.03 # Common default for LLMs
    warmup_steps = max(10, int(warmup_ratio * total_steps)) # Min 10 warmup steps
    
    save_steps_per_epoch = 1 
    save_steps = max(1, int(steps_per_epoch / save_steps_per_epoch))
    
    eval_times_per_epoch = min(5, max(1, steps_per_epoch // 20 if steps_per_epoch > 20 else 1)) 
    ideal_eval_steps = max(1, steps_per_epoch // eval_times_per_epoch)
    
    eval_steps = ideal_eval_steps
    # Ensure eval_steps is a divisor of save_steps if possible, or align them.
    # For simplicity, often eval_steps is just set independently or made equal to save_steps.
    # If save_steps is frequent, make eval_steps a multiple or equal.
    if save_steps > 0 and ideal_eval_steps > 0:
        if save_steps % ideal_eval_steps != 0:
            # Option 1: Make eval_steps a divisor of save_steps (could make eval infrequent)
            # for divisor in range(ideal_eval_steps, 0, -1):
            #     if save_steps % divisor == 0:
            #         eval_steps = divisor
            #         break
            # Option 2: Make eval_steps same as save_steps if close, or a reasonable fraction
            if ideal_eval_steps > save_steps / 2 : # If ideal eval is reasonably frequent
                 eval_steps = save_steps
            else: # If ideal eval is much more frequent than save, keep it more frequent
                 eval_steps = ideal_eval_steps
    elif ideal_eval_steps > 0 :
        eval_steps = ideal_eval_steps
    else:
        eval_steps = save_steps # Fallback

    eval_steps = max(1, eval_steps) # Ensure eval_steps is at least 1

    log_frequency_per_epoch = min(20, max(1, steps_per_epoch // 10 if steps_per_epoch > 10 else 1))
    logging_steps = max(1, steps_per_epoch // log_frequency_per_epoch)
    
    return {
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "save_steps": save_steps,
        "eval_steps": eval_steps,
        "logging_steps": logging_steps
    }

def update_config_file(config_path, batch_settings, step_settings, 
                      use_lora=False, use_qlora=False, quantization=None,
                      lora_rank=8, lora_alpha=16, lora_dropout=0.05,
                      precision="float16", gradient_checkpointing=True):
    """Update the training config YAML file with calculated settings"""
    
    # Backup original config
    backup_path = Path(str(config_path) + ".bak")
    try:
        shutil.copy2(config_path, backup_path)
        print(f"Backed up original config to: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not back up config file: {e}")

    with open(config_path, 'r') as f:
        # Using ruamel.yaml to preserve comments and structure if possible
        # For simplicity, stick to pyyaml if ruamel is not a standard dep for user
        try:
            import ruamel.yaml
            yaml_loader = ruamel.yaml.YAML()
            config = yaml_loader.load(f)
        except ImportError:
            print("ruamel.yaml not found, using PyYAML. Comments and formatting might be lost.")
            f.seek(0) # Reset file pointer for PyYAML
            config = yaml.safe_load(f)

    # Helper to ensure keys exist
    def ensure_key_path(conf, path_list, default_val=None):
        curr = conf
        for i, key in enumerate(path_list):
            if key not in curr:
                curr[key] = {} if i < len(path_list) - 1 else default_val
            curr = curr[key]
        return conf
    
    # Update batch settings
    ensure_key_path(config, ['model_args', 'batch_size'], batch_settings[0])
    config['model_args']['batch_size'] = batch_settings[0]
    ensure_key_path(config, ['train_args', 'gradient_accumulation_steps'], batch_settings[1])
    config['train_args']['gradient_accumulation_steps'] = batch_settings[1]
    
    # Update step settings
    ensure_key_path(config, ['train_args', 'max_steps'], step_settings['total_steps'])
    config['train_args']['max_steps'] = step_settings['total_steps']
    ensure_key_path(config, ['train_args', 'warmup_steps'], step_settings['warmup_steps'])
    config['train_args']['warmup_steps'] = step_settings['warmup_steps']
    ensure_key_path(config, ['train_args', 'save_steps'], step_settings['save_steps'])
    config['train_args']['save_steps'] = step_settings['save_steps']
    ensure_key_path(config, ['train_args', 'eval_steps'], step_settings['eval_steps'])
    config['train_args']['eval_steps'] = step_settings['eval_steps']
    ensure_key_path(config, ['train_args', 'logging_steps'], step_settings['logging_steps'])
    config['train_args']['logging_steps'] = step_settings['logging_steps']
    
    # Set precision
    ensure_key_path(config, ['train_args', 'fp16'], {'enabled': False})
    ensure_key_path(config, ['train_args', 'bf16'], {'enabled': False})
        
    if precision == "float16":
        config['train_args']['fp16']['enabled'] = True
        config['train_args']['bf16']['enabled'] = False
    elif precision == "bfloat16":
        config['train_args']['fp16']['enabled'] = False
        config['train_args']['bf16']['enabled'] = True
    else:  # float32
        config['train_args']['fp16']['enabled'] = False
        config['train_args']['bf16']['enabled'] = False
    
    # Set gradient checkpointing
    ensure_key_path(config, ['model_args', 'gradient_checkpointing'], gradient_checkpointing)
    config['model_args']['gradient_checkpointing'] = gradient_checkpointing
    
    # Update LoRA/QLoRA settings
    ensure_key_path(config, ['lora'], {})
    if use_lora or use_qlora:
        config['lora']['enable'] = True
        config['lora']['r'] = lora_rank
        config['lora']['alpha'] = lora_alpha
        config['lora']['dropout'] = lora_dropout
        config['lora']['target_modules'] = ["q_proj", "k_proj", "v_proj", "o_proj", 
                                           "gate_proj", "up_proj", "down_proj"] # Common defaults
        config['lora']['task_type'] = "CAUSAL_LM"
        
        if use_qlora:
            ensure_key_path(config, ['quantization'], {})
            config['quantization']['load_in_4bit'] = quantization == '4bit'
            config['quantization']['load_in_8bit'] = quantization == '8bit'
            if quantization == '4bit':
                config['quantization']['bnb_4bit_compute_dtype'] = precision
                config['quantization']['bnb_4bit_quant_type'] = "nf4"
                config['quantization']['bnb_4bit_use_double_quant'] = True
    elif 'lora' in config : # Explicitly disable if not selected
         config['lora']['enable'] = False


    with open(config_path, 'w') as f:
        if 'yaml_loader' in locals(): # Check if ruamel was used
             yaml_loader.dump(config, f)
        else:
             yaml.dump(config, f, default_flow_style=False, sort_keys=False) # Try to preserve order
    
    print(f"\nUpdated config file: {config_path}")
    # ... (rest of the print statements for summary)
    print(f"  Precision: {precision}, Grad Checkpointing: {gradient_checkpointing}")
    print(f"  Batch size: {batch_settings[0]}, Accumulation: {batch_settings[1]}, Effective Batch: {batch_settings[0] * batch_settings[1]}")
    print(f"  Total Steps: {step_settings['total_steps']}, Steps/Epoch: {step_settings['steps_per_epoch']}")
    print(f"  Warmup: {step_settings['warmup_steps']}, Save: {step_settings['save_steps']}, Eval: {step_settings['eval_steps']}, Log: {step_settings['logging_steps']}")
    if use_lora or use_qlora:
        peft_method = "QLoRA" if use_qlora else "LoRA"
        quant_str = f" with {quantization} quantization" if use_qlora else ""
        print(f"  Using {peft_method}{quant_str} (rank={lora_rank}, alpha={lora_alpha})")
    
    print(f"\nReminder: If effective batch size ({batch_settings[0] * batch_settings[1]}) has changed significantly,")
    print(f"  you might need to adjust the learning rate. A common heuristic is to scale it")
    print(f"  linearly or with a square root rule relative to the original effective batch size.")


def estimate_model_size_from_params(num_total_params_billions, precision_for_storage="float32"):
    """Estimate model size in GB based on number of parameters, assuming parameters are stored at a certain precision."""
    bytes_per_param = 4
    if precision_for_storage == "float16" or precision_for_storage == "bfloat16":
        bytes_per_param = 2
    elif precision_for_storage == "int8": # For 8-bit quantized models base size
        bytes_per_param = 1

    # num_total_params_billions is in billions (e.g., 7 for 7B)
    size_gb = (num_total_params_billions * 1e9 * bytes_per_param) / (1024**3)
    return size_gb

def get_model_arch_defaults(model_params_billions):
    """Provide rough defaults for hidden_size and num_layers based on model size (Llama-like)."""
    if model_params_billions <= 1.5: # ~1B models
        return 2048, 24 # e.g. Llama 1B type
    elif model_params_billions <= 3: # ~3B models
        return 3200, 26 # e.g. Llama 3 8B uses 4096,32 - this is smaller
    elif model_params_billions <= 7.5: # ~7B models
        return 4096, 32 # e.g. Llama 7B
    elif model_params_billions <= 13.5: # ~13B models
        return 5120, 40 # e.g. Llama 13B
    elif model_params_billions <= 35: # ~30-35B models
        return 6656, 60 # Approximation
    elif model_params_billions <= 75: # ~70B models
        return 8192, 80 # e.g. Llama 70B
    else: # Larger models
        return 10240, 96 # Generic large
        
def main():
    parser = argparse.ArgumentParser(
        description="Automatically configure training parameters based on GPU memory and model size.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    model_group = parser.add_argument_group('Model Architecture & Size')
    model_group.add_argument("--model_params_billions", type=float, required=True, 
                             help="Total model parameters in billions (e.g., 7 for a 7B model). Used to estimate FP32 size and arch defaults.")
    model_group.add_argument("--model_hidden_size", type=int, 
                             help="Model hidden size (embedding dimension). If not provided, estimated from --model_params_billions.")
    model_group.add_argument("--num_model_layers", type=int,
                             help="Number of transformer layers in the model. If not provided, estimated from --model_params_billions.")

    # Config and Dataset
    config_group = parser.add_argument_group('Configuration & Dataset')
    config_group.add_argument("--config", type=str, required=True, help="Path to training config.yaml file to update.")
    config_group.add_argument("--dataset_size", type=int, required=True, help="Number of training examples in the dataset.")
    
    # Training parameters
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    train_group.add_argument("--seq_length", type=int, default=2048, help="Maximum sequence length during training.")
    train_group.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
                             help="Training precision. Defaults to bfloat16 if supported, else float16.")
    train_group.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True,
                             help="Enable gradient checkpointing to save memory.")
    
    # PEFT options
    peft_group = parser.add_argument_group('PEFT (LoRA/QLoRA) Parameters')
    peft_group.add_argument("--lora", action=argparse.BooleanOptionalAction, default=False, help="Use LoRA for training.")
    peft_group.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=False, help="Use QLoRA for training (implies LoRA).")
    peft_group.add_argument("--quantization", choices=["4bit", "8bit"], default="4bit", 
                            help="Quantization precision for QLoRA.")
    peft_group.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (r).") # Increased default
    peft_group.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha. Often 2*rank.") # Increased default
    peft_group.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability.")
    
    # System parameters
    sys_group = parser.add_argument_group('System & Dataloader Parameters')
    sys_group.add_argument("--dataloader_workers", type=int, default=2, help="Number of dataloader workers for prefetching.")
    sys_group.add_argument("--system_reserve_gpu_gb", type=float, default=1.5, # Slightly increased default
                           help="GPU memory (GB) to reserve for system, CUDA overhead, and non-PyTorch processes.")
    
    args = parser.parse_args()
    
    # --- Argument Validation and Processing ---
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return

    if args.qlora: # QLoRA implies LoRA
        args.lora = True 
    if args.lora and args.lora_alpha is None: # Default alpha if LoRA is used
        args.lora_alpha = 2 * args.lora_rank
        print(f"Defaulting LoRA alpha to 2*rank = {args.lora_alpha}")

    # Estimate model architecture if not provided
    default_hidden_size, default_num_layers = get_model_arch_defaults(args.model_params_billions)
    model_hidden_size = args.model_hidden_size if args.model_hidden_size is not None else default_hidden_size
    num_model_layers = args.num_model_layers if args.num_model_layers is not None else default_num_layers
    print(f"Using Model Architecture: Hidden Size = {model_hidden_size}, Num Layers = {num_model_layers}")

    # Estimate base model size in FP32 (used for relative calculations)
    model_size_gb_fp32 = estimate_model_size_from_params(args.model_params_billions, "float32")
    print(f"Estimated base model size (FP32 equivalent): {model_size_gb_fp32:.2f} GB from {args.model_params_billions}B parameters.")

    # --- Memory Info ---
    gpu_memory = get_gpu_memory_info()
    if gpu_memory["total"] == 0:
        print("Warning: No GPU detected or CUDA not available. Calculations might be inaccurate for CPU mode.")
        available_gpu_for_pytorch = 0
    else:
        available_gpu_for_pytorch = gpu_memory['available_for_new_pytorch_alloc'] 
        print(f"GPU Memory Info: Total={gpu_memory['total']:.2f}GB, PyTorch Reserved={gpu_memory['reserved_pytorch']:.2f}GB, Available for new PyTorch alloc={available_gpu_for_pytorch:.2f}GB (Raw Free by OS: {gpu_memory['raw_available']:.2f}GB)")

    ram_memory = get_ram_memory_info()
    print(f"RAM Memory Info: Total={ram_memory['total']:.2f}GB, Available={ram_memory['available']:.2f}GB")
    dataset_ram_info = calculate_dataset_memory(args.dataset_size, args.seq_length, 1, args.dataloader_workers) # Batch size 1 for estimation
    print(f"Estimated RAM impact for dataset handling/prefetching (approx): {dataset_ram_info['ram_impact_gb']:.2f} GB")
    if dataset_ram_info['ram_impact_gb'] > ram_memory['available'] * 0.5:
        print("Warning: Estimated dataset RAM impact is significant compared to available system RAM.")

    # --- Calculations ---
    batch_size, accumulation_steps = calculate_batch_settings(
        model_size_gb_fp32, 
        available_gpu_for_pytorch,
        seq_length=args.seq_length,
        model_hidden_size=model_hidden_size,
        num_model_layers=num_model_layers,
        use_lora=args.lora,
        use_qlora=args.qlora,
        quantization=args.quantization,
        lora_rank=args.lora_rank,
        precision=args.precision,
        dataset_size=args.dataset_size,
        dataloader_workers=args.dataloader_workers,
        system_reserve_gpu_gb=args.system_reserve_gpu_gb,
        gradient_checkpointing=args.gradient_checkpointing
    )
    
    step_settings = calculate_training_steps(
        args.dataset_size, batch_size, accumulation_steps, args.epochs
    )
    
    print("="*50+"\nStep Settings\n"+"="*50)
    print(step_settings)
    print("="*100)
    
    # --- Update Config File ---
    # if step_settings["total_steps"] > 0 : # Only update if steps are valid
    #     update_config_file(
    #         config_path, 
    #         (batch_size, accumulation_steps), 
    #         step_settings,
    #         use_lora=args.lora, use_qlora=args.qlora, quantization=args.quantization,
    #         lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    #         precision=args.precision, gradient_checkpointing=args.gradient_checkpointing
    #     )
    #     print("\nConfiguration updated successfully!")
    # else:
    #     print("\nSkipping configuration file update due to invalid step calculation (total_steps is 0).")
    #     print("This usually means the effective batch size is too large for the dataset size and number of epochs.")
    #     print("Please check your dataset_size, batch_size, accumulation_steps, and epochs.")


if __name__ == "__main__":
    main()