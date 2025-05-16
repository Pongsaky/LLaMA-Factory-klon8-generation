import os
import yaml
import torch
import argparse
import numpy as np
import psutil
import shutil # For file backup
from pathlib import Path

# NEW: Import AutoConfig from transformers
try:
    from transformers import AutoConfig, PretrainedConfig
except ImportError:
    print("Hugging Face `transformers` library not found. Please install it: pip install transformers")
    exit(1)

# --- GPU Profiles (VRAM in GB) ---
# Add more profiles as needed. These are approximate and can vary slightly by manufacturer.
GPU_PROFILES = {
    "local": None, # Special value to detect local GPU
    # NVIDIA Data Center GPUs
    "A100-40GB": 40.0,
    "A100-80GB": 80.0,
    "H100-80GB": 80.0,
    "V100-16GB": 16.0,
    "V100-32GB": 32.0,
    "T4-16GB": 16.0, # Actually 15GB usable for some
    # NVIDIA RTX / GeForce (Consumer/Prosumer)
    "RTX4090-24GB": 24.0,
    "RTX3090-24GB": 24.0,
    "RTX3080-10GB": 10.0,
    "RTX3080-12GB": 12.0,
    "RTXA6000-48GB": 48.0,
    "RTXA5000-24GB": 24.0,
    # AMD GPUs (example, actual VRAM may vary)
    "MI250X-128GB": 128.0, # Per GCD, so often used as 64GB
    "MI210-64GB": 64.0,
    # CPU mode (for testing script logic without GPU)
    "CPU-lowRAM": 8.0, # Simulate low resource for CPU
    "CPU-medRAM": 16.0,
    "CPU-highRAM": 32.0,
}

def get_hf_model_architecture(model_name_or_path: str):
    """
    Fetches model configuration from Hugging Face Hub and extracts architecture details.
    Returns a dictionary with 'num_hidden_layers', 'hidden_size', 'num_attention_heads', 
    'vocab_size', and 'estimated_total_params_billions'.
    """
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        details = {
            "num_hidden_layers": None,
            "hidden_size": None,
            "num_attention_heads": None,
            "vocab_size": None,
            "estimated_total_params_billions": None,
            "model_type": config.model_type if hasattr(config, "model_type") else "unknown"
        }

        # Common attribute names for number of layers
        layer_attrs = ["num_hidden_layers", "n_layer", "num_layers"]
        for attr in layer_attrs:
            if hasattr(config, attr):
                details["num_hidden_layers"] = getattr(config, attr)
                break
        
        # Common attribute names for hidden size
        hidden_size_attrs = ["hidden_size", "n_embd", "d_model"]
        for attr in hidden_size_attrs:
            if hasattr(config, attr):
                details["hidden_size"] = getattr(config, attr)
                break

        # Common attribute names for attention heads
        head_attrs = ["num_attention_heads", "n_head", "num_heads"]
        for attr in head_attrs:
            if hasattr(config, attr):
                details["num_attention_heads"] = getattr(config, attr)
                break
        
        if hasattr(config, "vocab_size"):
            details["vocab_size"] = config.vocab_size

        # Estimate total parameters if not directly available (very rough)
        # This is a fallback; better if the user knows or it's in a more structured place.
        # For many models, params ~ vocab_size * hidden_size (embeddings) + num_layers * (complex_terms_involving_hidden_size_sq)
        if details["num_hidden_layers"] and details["hidden_size"] and details["vocab_size"]:
            # Simplified Llama-like estimation (very approximate)
            # Embedding layer params
            embedding_params = details["vocab_size"] * details["hidden_size"]
            # Attention params per layer (Q,K,V,O projections)
            # Each is roughly hidden_size * hidden_size. So 4 * h^2.
            attention_params_per_layer = 4 * (details["hidden_size"] ** 2)
            # MLP params per layer (gate, up, down projections)
            # Often MLP hidden is 2.66x to 4x hidden_size (e.g. SwiGLU for Llama uses ~2.66 * (2/3) * 2 for up/gate, and down)
            # (intermediate_size * hidden_size) * 2 for FFN layers (approx)
            # or for SwiGLU-like: ( ( ( (hidden_size * 4 * 2 // 3) + 255 ) // 256 * 256) * hidden_size ) * 2
            mlp_intermediate_size = getattr(config, "intermediate_size", int(details["hidden_size"] * 3.5)) # Common ratio
            mlp_params_per_layer = mlp_intermediate_size * details["hidden_size"] + \
                                   mlp_intermediate_size * details["hidden_size"] # Up and Down projection
            
            transformer_block_params = details["num_hidden_layers"] * (attention_params_per_layer + mlp_params_per_layer)
            # Add layernorm params (small), output projection (hidden_size * vocab_size, if not tied)
            # This estimation is still very rough.
            # A PretrainedConfig object might have a `num_parameters` method or attribute in some versions/models
            # but it's not consistently available.
            
            # If model has a num_parameters method (some newer configs might)
            if hasattr(config, "num_parameters"):
                 total_params = config.num_parameters(exclude_embeddings=False) # Fictional method, for illustration
                 # Actual method might be different or non-existent
            else: # Fallback to our rough estimation
                 total_params = embedding_params + transformer_block_params 
                 # Add output embedding if not tied
                 if not getattr(config, "tie_word_embeddings", True):
                     total_params += details["vocab_size"] * details["hidden_size"]

            details["estimated_total_params_billions"] = total_params / 1e9
            
            if not all([details["num_hidden_layers"], details["hidden_size"]]):
                 print(f"Warning: Could not reliably determine hidden_size or num_layers for {model_name_or_path}.")

        print(f"Fetched config for model type: {details['model_type']}")
        return details

    except Exception as e:
        print(f"Error fetching or parsing model config for '{model_name_or_path}': {e}")
        print("Please ensure the model name is correct and accessible, or provide parameters manually if necessary.")
        return None

def get_gpu_memory_info(target_gpu_profile_name: str = "local"):
    """Returns detailed GPU memory information in GB, using profile if specified."""
    
    if target_gpu_profile_name != "local" and target_gpu_profile_name in GPU_PROFILES:
        profile_vram = GPU_PROFILES[target_gpu_profile_name]
        if profile_vram is None: # Should not happen if "local" is handled
            print(f"Error: Profile '{target_gpu_profile_name}' has no VRAM defined, falling back to local detection.")
        else:
            print(f"Using VRAM from target GPU profile '{target_gpu_profile_name}': {profile_vram:.2f} GB")
            # For profiled GPU, assume all of it is "available for new PyTorch alloc" for planning purposes,
            # as we don't know its current reserved/allocated state.
            return {
                "total": profile_vram,
                "reserved_pytorch": 0.0, # Unknown for remote/profiled target
                "allocated_pytorch": 0.0,  # Unknown for remote/profiled target
                "available_for_new_pytorch_alloc": profile_vram, 
                "raw_available": profile_vram, # Best guess for profiled
                "is_profiled": True
            }

    # Fallback to local detection
    if not torch.cuda.is_available():
        print("CUDA not available on this machine, using CPU mode for local VRAM (0 GB effective).")
        # If user chose a CPU profile, that VRAM is already returned above.
        # This case is for --target_gpu_profile local on a CPU machine.
        cpu_vram = GPU_PROFILES.get(target_gpu_profile_name, 0.0) if "CPU" in target_gpu_profile_name else 0.0
        return {
            "total": cpu_vram, "reserved_pytorch": 0.0, "allocated_pytorch": 0.0,
            "available_for_new_pytorch_alloc": cpu_vram, "raw_available": cpu_vram, "is_profiled": "CPU" in target_gpu_profile_name
        }
    
    gpu_id = 0 
    torch.cuda.empty_cache() # Try to clear cache for more accurate reading
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
    reserved_cuda_memory = torch.cuda.memory_reserved(gpu_id) / (1024**3)
    allocated_cuda_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
    
    # Free memory as reported by nvidia-smi like tools (total_free_device_memory)
    # This is often what users "see" as available before PyTorch starts reserving.
    raw_driver_free_memory = torch.cuda.mem_get_info()[0] / (1024**3)

    # PyTorch available = total_memory - reserved_by_pytorch_caching_allocator
    # This is memory PyTorch *could* allocate without needing more from OS/driver.
    # However, for planning new large allocations, raw_driver_free_memory is often a better indicator
    # of what's truly free on the device, before PyTorch grabs a large chunk for its cache.
    # Let's use the more conservative of (Total - PyTorch Reserved) and (Raw Driver Free)
    # available_for_new_allocations = min(total_memory - reserved_cuda_memory, raw_driver_free_memory)
    # A common Pytorch pattern is that it reserves a large pool. 
    # So, total_memory - reserved_cuda_memory is how much Pytorch *thinks* it has left in its pool.
    # if reserved_cuda_memory is very small (nothing cached yet), then this is close to total_memory.
    # In this case, raw_driver_free_memory is more realistic.
    
    # If PyTorch has already reserved a significant chunk, then `total_memory - reserved_cuda_memory` is what's left in that pool.
    # If PyTorch hasn't reserved much, then `raw_driver_free_memory` is what's truly available on the device.
    # The actual memory PyTorch can end up using for *new, large* allocations is closer to `raw_driver_free_memory`
    # minus what it *already* has allocated for tensors.
    
    available_for_new_large_pytorch_alloc = raw_driver_free_memory # Start with what driver says is free
                                                               # PyTorch will then manage this.
    
    return {
        "total": total_memory,
        "reserved_pytorch": reserved_cuda_memory, 
        "allocated_pytorch": allocated_cuda_memory, 
        "available_for_new_pytorch_alloc": available_for_new_large_pytorch_alloc, 
        "raw_available": raw_driver_free_memory,
        "is_profiled": False
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

def calculate_model_memory(
    model_params_billions, model_hidden_size, num_model_layers, vocab_size, # Added vocab_size
    precision="float16", use_lora=False, use_qlora=False, quantization=None, 
    lora_rank=8, tie_word_embeddings=True, model_type="unknown" # Added tie_word_embeddings
    ):
    """
    Calculate static model memory requirements in GB.
    model_params_billions: Total parameters of the base model in billions.
    """
    bytes_per_param_fp32 = 4.0
    model_size_gb_fp32 = model_params_billions * bytes_per_param_fp32 # Reference FP32 size

    # 1. Model Weights Memory
    if use_qlora:
        quant_factor = 0.125 if quantization == '4bit' else 0.25
        model_weights_memory = model_size_gb_fp32 * quant_factor
        print(f"  Model weights (QLoRA {quantization}): {model_weights_memory:.2f} GB (from {model_params_billions:.2f}B params at FP32 ref)")
    else: # Full model or LoRA on unquantized base
        precision_factor = 1.0 if precision == "float32" else 0.5 # For fp16/bf16
        model_weights_memory = model_size_gb_fp32 * precision_factor
        print(f"  Model weights ({precision}): {model_weights_memory:.2f} GB (from {model_params_billions:.2f}B params at FP32 ref)")

    # 2. Optimizer States Memory
    # AdamW needs 2 states (m, v) per trainable parameter.
    # Each state is usually same precision as training, or FP32. Assume same as training precision for estimation.
    bytes_per_optimizer_param_state = 2.0 if precision in ["float16", "bfloat16"] else 4.0
    
    if use_lora or use_qlora:
        # Estimate LoRA trainable parameters more accurately if possible
        # For a typical dense layer: params_in = H_in * R, params_out = R * H_out
        # For Llama-like models, LoRA is often applied to q,k,v,o projections and sometimes gate/up/down in MLP.
        # Number of LoRA-fied layers: num_model_layers
        # Per LoRA-fied linear layer (e.g., q_proj): hidden_size * lora_rank (A) + lora_rank * hidden_size (B)
        # Let's assume 4 such projections in attention + 2-3 in MLP per layer are targeted.
        num_lora_targets_per_block = 6 # q,k,v,o and 2 MLP layers (gate_proj, down_proj, up_proj etc.)
        
        # This is a simplification, as not all targets have same input/output dims matching hidden_size always
        # but hidden_size is a dominant factor.
        num_lora_params_approx = num_model_layers * num_lora_targets_per_block * (2 * model_hidden_size * lora_rank)
        
        optimizer_trainable_params_gb = (num_lora_params_approx * bytes_per_optimizer_param_state * 2) / (1024**3)
        print(f"  LoRA trainable params (approx): {num_lora_params_approx/1e6:.2f}M")
    else: # Full fine-tuning
        num_full_model_params_approx = model_params_billions * 1e9
        optimizer_trainable_params_gb = (num_full_model_params_approx * bytes_per_optimizer_param_state * 2) / (1024**3)
    
    optimizer_memory = optimizer_trainable_params_gb
    print(f"  Optimizer states ({precision}, for trainable params): {optimizer_memory:.2f} GB")

    # 3. Static Activation Memory (portion independent of batch size, related to model structure, minimal with grad checkpointing)
    # This is very hard to estimate accurately, so we use a small heuristic overhead.
    # The bulk of activation memory is batch-dependent and handled in `calculate_batch_memory_per_sample`.
    # This accounts for some framework overhead for activations, irreducible parts.
    static_activation_overhead_gb = model_weights_memory * 0.1 # 10% of weights as a small static part
    
    # Add a general fixed overhead for miscellaneous framework needs, CUDA context, etc.
    # This is part of the script's "system_reserve_gpu_gb" but a small part can be attributed to model loading itself.
    framework_fixed_overhead_gb = 0.5 # Fixed 0.5 GB for framework basics around the model
    
    total_static_model_memory = model_weights_memory + optimizer_memory + static_activation_overhead_gb + framework_fixed_overhead_gb
    
    # General very conservative overhead on top of all calculated static parts
    conservative_static_overhead_factor = 1.15 # Increased to 15%
    total_static_model_memory_with_overhead = total_static_model_memory * conservative_static_overhead_factor
    
    print(f"  Static activation/framework overhead (approx): {static_activation_overhead_gb + framework_fixed_overhead_gb:.2f} GB")
    print(f"  Subtotal static model memory: {total_static_model_memory:.2f} GB")
    print(f"  Total static model memory (with { (conservative_static_overhead_factor-1)*100:.0f}% overhead): {total_static_model_memory_with_overhead:.2f} GB")
    
    return {
        "model_weights": model_weights_memory,
        "optimizer_states": optimizer_memory,
        "static_activation_framework_overhead": static_activation_overhead_gb + framework_fixed_overhead_gb,
        "total_static_model_memory_with_overhead": total_static_model_memory_with_overhead
    }

def calculate_dataset_memory_ram(dataset_size, seq_length, effective_batch_size, dataloader_workers=2):
    bytes_per_token_id = 4 # int32 for token IDs
    # input_ids, attention_mask, labels. Labels might not always be seq_length.
    elements_per_sample = 2.5 # Avg 2.5 full sequences of token IDs
    bytes_per_sample_ram = seq_length * bytes_per_token_id * elements_per_sample
    
    # Dataloader prefetching in RAM: workers + main process might hold a batch
    prefetch_factor = dataloader_workers + 2 
    prefetched_ram_gb = (effective_batch_size * prefetch_factor * bytes_per_sample_ram) / (1024**3)
    
    # General overhead for dataset objects, tokenizers in RAM
    dataset_obj_ram_gb = 0.5 + (dataset_size * 100 / (1024**3)) # 100 bytes per sample metadata approx
    
    total_dataset_ram_impact_gb = prefetched_ram_gb + dataset_obj_ram_gb
    return {"ram_impact_gb": total_dataset_ram_impact_gb}

def calculate_batch_memory_per_sample_gpu(
    seq_length, model_hidden_size, num_model_layers, num_attention_heads, vocab_size,
    precision="float16", gradient_checkpointing=True, model_type="unknown"
    ):
    """Estimates GPU memory PER SAMPLE in a batch for dynamic components (activations, KV cache, gradients)."""
    bytes_per_element = 2.0 if precision in ["float16", "bfloat16"] else 4.0

    # 1. Activations per sample (dynamic part)
    # Formula from https://huggingface.co/docs/transformers/perf_train_gpu_one#anatomy-of-model-memory
    # Simplified: S * B * H * L * (factor)
    # For transformers, activations are roughly:
    # seq_len * hidden_size * num_layers * (various small factors for different types of activations)
    # With gradient checkpointing, this is significantly reduced. The formula often cited is
    # proportional to sqrt(num_layers) instead of num_layers for the largest activation checkpoint.
    # Let's use a heuristic based on common observations.
    # Size of largest activation tensor: seq_len * hidden_size * bytes_per_element
    # Number of such large tensors stored if checkpointing: ~1-2 per block, related to sqrt(num_layers)
    # If not checkpointing: proportional to num_layers.
    
    activation_factor_heuristic = 20 # Base factor from HF guide for some configs
    if gradient_checkpointing:
        # Greatly reduces activations, but not to zero. Some are still needed.
        # Small per-layer activations + sqrt(L) blocks of larger ones.
        # Heuristic: Let's assume it's like a few full layers' worth of activations.
        activation_memory_per_sample = seq_length * model_hidden_size * bytes_per_element * (num_model_layers**0.5) * 0.5 
    else:
        # Rough estimate: sum of activations across layers. Assume ~10-20x one layer's main activation map.
        # This needs to cover all intermediate activations in a layer (after attention, after MLP).
        activation_memory_per_sample = seq_length * model_hidden_size * num_model_layers * bytes_per_element * activation_factor_heuristic * 0.1 # Scaled down

    # 2. KV Cache per sample (only during inference, but some frameworks might pre-allocate if batch_size is fixed)
    # For training, usually not a persistent KV cache in the same way unless specific optimizations are used.
    # However, attention mechanisms do compute K, V, and scores which are batch_size dependent.
    # Let's consider the memory for K, V, and attention scores.
    # K, V: seq_len * hidden_size * bytes_per_element * 2 (for K and V) * num_layers (if all cached, not typical in training)
    # Attention scores: seq_len * seq_len * num_attention_heads * bytes_per_element (potentially large!)
    # For training, these are mostly transient.
    # Let's focus on the peak memory for one attention layer's computation.
    # QK^T part: batch_size * num_heads * seq_len * seq_len
    # If seq_len is large, this dominates.
    # This is very complex. Sticking to simpler HF guide numbers:
    # Add a component for transient attention calculation memory per sample.
    attention_transient_per_sample = (num_attention_heads * seq_length**2 * bytes_per_element) * 0.1 # Small fraction as it's transient for one layer at a time mostly

    # 3. Gradients for activations (if not recomputed by grad checkpointing)
    # Similar size to activations if they are stored. Reduced by grad checkpointing.
    activation_gradients_per_sample = activation_memory_per_sample if not gradient_checkpointing else activation_memory_per_sample * 0.5 # Smaller if recomputed

    # 4. Raw batch data on GPU (input_ids, attention_mask, labels)
    bytes_per_token_id_gpu = 4 # int32 or int64
    raw_input_data_per_sample = seq_length * bytes_per_token_id_gpu * 2.5 # input_ids, attention_mask, labels (approx)

    total_dynamic_per_sample_bytes = (
        activation_memory_per_sample +
        attention_transient_per_sample + # This might be an overestimate or double count with activation part
        activation_gradients_per_sample +
        raw_input_data_per_sample
    )
    
    # Add a conservative overhead factor for this dynamic part too
    dynamic_overhead_factor = 1.20 # 20% overhead on dynamic per-sample calculations
    total_dynamic_per_sample_gb = (total_dynamic_per_sample_bytes * dynamic_overhead_factor) / (1024**3)
    
    # print(f"    Per-sample dynamic breakdown (GB): Act={activation_memory_per_sample/(1024**3):.4f}, AttnTrans={attention_transient_per_sample/(1024**3):.4f}, ActGrad={activation_gradients_per_sample/(1024**3):.4f}, RawData={raw_input_data_per_sample/(1024**3):.4f}")
    return total_dynamic_per_sample_gb

def calculate_batch_settings(
    available_gpu_memory_gb, model_arch, # model_arch is dict from get_hf_model_architecture
    seq_length=2048, 
    use_lora=False, use_qlora=False, quantization=None, lora_rank=8,
    precision="float16", 
    system_reserve_gpu_gb=2.0, # Increased default
    gradient_checkpointing=True
):
    model_params_billions = model_arch["estimated_total_params_billions"]
    model_hidden_size = model_arch["hidden_size"]
    num_model_layers = model_arch["num_hidden_layers"]
    num_attention_heads = model_arch["num_attention_heads"]
    vocab_size = model_arch["vocab_size"]
    model_type = model_arch["model_type"]

    if None in [model_params_billions, model_hidden_size, num_model_layers, num_attention_heads, vocab_size]:
        print("Error: Critical model architecture details are missing. Cannot proceed with batch calculation.")
        print(f"  Details received: Params={model_params_billions}B, Hidden={model_hidden_size}, Layers={num_model_layers}, Heads={num_attention_heads}, Vocab={vocab_size}")
        return 1, 64 # Fallback

    print(f"\nCalculating batch settings for Model: {model_type}, {model_params_billions:.2f}B params, {num_model_layers} layers, {model_hidden_size} hidden dim.")
    print(f"  Training Precision: {precision}, Grad Checkpointing: {gradient_checkpointing}")
    if use_qlora: print(f"  QLoRA: Enabled, Quantization: {quantization}, Rank: {lora_rank}")
    elif use_lora: print(f"  LoRA: Enabled, Rank: {lora_rank}")

    static_model_mem_info = calculate_model_memory(
        model_params_billions, model_hidden_size, num_model_layers, vocab_size,
        precision=precision, use_lora=use_lora, use_qlora=use_qlora, 
        quantization=quantization, lora_rank=lora_rank, model_type=model_type
    )
    
    static_model_memory_needed_gb = static_model_mem_info["total_static_model_memory_with_overhead"]
    
    memory_for_dynamic_batch_parts_gb = available_gpu_memory_gb - static_model_memory_needed_gb - system_reserve_gpu_gb
    
    print(f"\nGPU Memory Allocation for Batches:")
    print(f"  Available GPU (for PyTorch new alloc): {available_gpu_memory_gb:.2f} GB")
    print(f"  (-) Static Model Memory (loaded model, optimizer states): {static_model_memory_needed_gb:.2f} GB")
    print(f"  (-) System & Safety Reserve on GPU: {system_reserve_gpu_gb:.2f} GB")
    print(f"  (=) Memory Remaining for Dynamic Batch Parts: {memory_for_dynamic_batch_parts_gb:.2f} GB")

    if memory_for_dynamic_batch_parts_gb <= 0.1: # Need at least a small amount
        print(f"\nCritical Warning: Not enough memory for dynamic batch parts after accounting for static model and system reserve!")
        print(f"  Available for dynamic parts: {memory_for_dynamic_batch_parts_gb:.2f} GB. This is too low.")
        # (Further suggestions as before)
        print(f"Suggestions to reduce memory:")
        print(f"  - Ensure '--target_gpu_profile' (if used) matches a GPU with sufficient VRAM.")
        print(f"  - If running locally, close other GPU-intensive applications.")
        print(f"  - Use QLoRA with 4-bit: --qlora --quantization 4bit (if not already).")
        print(f"  - Enable gradient checkpointing: --gradient_checkpointing (if not already).")
        print(f"  - Reduce sequence length: --seq_length (current: {seq_length}).")
        print(f"  - Reduce LoRA rank: --lora_rank (current: {lora_rank}) if using LoRA/QLoRA.")
        print(f"  - Increase system reserve '--system_reserve_gpu_gb' if there's unexpected external GPU usage.")
        return 1, 128 # Fallback: Minimum batch size, very high accumulation

    # Calculate how much memory one sample (sequence) in a batch would consume dynamically
    dynamic_memory_per_sample_gb = calculate_batch_memory_per_sample_gpu(
        seq_length, model_hidden_size, num_model_layers, num_attention_heads, vocab_size,
        precision, gradient_checkpointing, model_type
    )
    print(f"  Estimated dynamic GPU memory per sample in batch (activations, gradients, data): {dynamic_memory_per_sample_gb:.4f} GB")

    if dynamic_memory_per_sample_gb <= 1e-6 : # Should be positive
        print("Error: Calculated dynamic memory per sample is too low or zero. Check model architecture parameters or calculation logic.")
        return 1, 128 

    max_theoretical_batch_size = int(memory_for_dynamic_batch_parts_gb / dynamic_memory_per_sample_gb)
    
    if max_theoretical_batch_size < 1:
        print(f"Warning: Max theoretical batch size is < 1 ({max_theoretical_batch_size}). Not enough memory for even a single sample's dynamic parts.")
        # (Further suggestions as above)
        return 1, 128

    # Determine a practical batch size (heuristic, power of 2 often not strictly needed but common)
    # More conservative now: if max_theoretical_batch_size is e.g. 5, use 4.
    if max_theoretical_batch_size >= 64: batch_size = 32 # Cap initial practical at 32 for stability/generality
    elif max_theoretical_batch_size >= 32: batch_size = 16
    elif max_theoretical_batch_size >= 16: batch_size = 8
    elif max_theoretical_batch_size >= 8: batch_size = 4
    elif max_theoretical_batch_size >= 4: batch_size = 2
    elif max_theoretical_batch_size >= 2: batch_size = 2 # Prefer 2 over 1 if possible
    else: batch_size = 1
    
    batch_size = max(1, min(batch_size, max_theoretical_batch_size)) # Ensure it's within feasible range and at least 1.

    # Target effective batch size (can be a hyperparameter)
    target_effective_batch = 64 
    if use_qlora or use_lora: target_effective_batch = 128 # PEFT can often handle larger effective batch sizes
    
    # Adjust target based on available total VRAM for very small GPUs
    if available_gpu_memory_gb < 12: target_effective_batch = max(32, target_effective_batch // 2)
    if available_gpu_memory_gb < 8: target_effective_batch = max(16, target_effective_batch // 2)

    accumulation_steps = max(1, int(np.ceil(target_effective_batch / batch_size)))
    final_effective_batch_size = batch_size * accumulation_steps
    
    print(f"\nBatch Settings Calculation:")
    print(f"  Max theoretical batch size (fitting dynamic parts): {max_theoretical_batch_size}")
    print(f"  Selected practical batch_size (fitting dynamic parts): {batch_size}")
    print(f"  Target effective batch_size: {target_effective_batch}")
    print(f"  Calculated gradient_accumulation_steps: {accumulation_steps}")
    print(f"  Final effective batch_size: {final_effective_batch_size}")

    if final_effective_batch_size < 8 and not (use_lora or use_qlora): # Small effective batch for full FT can be tricky
        print("Warning: Effective batch size is very small for full fine-tuning. Training might be unstable. Consider more accumulation or smaller target.")
    elif final_effective_batch_size < 4: # Generally very small
        print("Warning: Effective batch size is very small (<4). Training might be unstable.")

    return batch_size, accumulation_steps

def calculate_training_steps(total_samples, batch_size, accumulation_steps, epochs):
    if batch_size <= 0 or accumulation_steps <= 0:
        print("Error: Batch size or accumulation steps is zero or negative, cannot calculate training steps.")
        return { "total_steps": 0, "steps_per_epoch": 0, "warmup_steps": 0, "save_steps": 0, "eval_steps": 0, "logging_steps": 0 }
        
    effective_batch_size = batch_size * accumulation_steps
    if effective_batch_size == 0: # Should be caught by above, but defensive
        print("Error: Effective batch size is zero.")
        return { "total_steps": 0, "steps_per_epoch": 0, "warmup_steps": 0, "save_steps": 0, "eval_steps": 0, "logging_steps": 0 }

    steps_per_epoch = total_samples // effective_batch_size
    if steps_per_epoch == 0:
        print(f"Warning: Calculated steps_per_epoch is 0 (Dataset: {total_samples}, Effective Batch: {effective_batch_size}). Effective batch size may be larger than dataset size for one epoch.")
        print("  This implies the model will see all data in less than one nominal 'epoch'.")
        print("  Setting steps_per_epoch to 1. Consider if epochs parameter is meaningful or if max_steps is a better target.")
        steps_per_epoch = 1 
        
    total_steps = steps_per_epoch * epochs
    warmup_ratio = 0.03 
    warmup_steps = max(10, int(warmup_ratio * total_steps))
    
    # Save/Eval/Log steps based on steps_per_epoch
    # Aim for 1-2 saves per epoch, 2-5 evals, ~10-20 logs.
    save_divisor = max(1, steps_per_epoch // 2 if steps_per_epoch > 1 else 1) # Save 1-2 times per epoch
    save_steps = max(1, save_divisor)

    eval_divisor = max(1, steps_per_epoch // 5 if steps_per_epoch > 4 else 1) # Eval up to 5 times per epoch
    eval_steps = max(1, eval_divisor)
    # Align eval with save if they are close, or make eval more frequent.
    if save_steps > 0 and eval_steps > 0 and save_steps != eval_steps:
        if abs(save_steps - eval_steps) < eval_steps * 0.5 : # If they are close
            eval_steps = save_steps 
        elif eval_steps > save_steps: # Ensure eval is not less frequent than save
                 eval_steps = save_steps
    eval_steps = max(1, eval_steps)


    log_divisor = max(1, steps_per_epoch // 20 if steps_per_epoch > 19 else 1) # Log ~20 times per epoch
    logging_steps = max(1, log_divisor)
    # Ensure logging is at least as frequent as eval
    if logging_steps > eval_steps and eval_steps > 0 : logging_steps = eval_steps
    logging_steps = max(1, min(logging_steps, 50) if total_steps < 1000 else 100) # Cap logging frequency for very short/long runs
    
    return {
        "total_steps": total_steps, "steps_per_epoch": steps_per_epoch, "warmup_steps": warmup_steps,
        "save_steps": save_steps, "eval_steps": eval_steps, "logging_steps": logging_steps
    }

def update_config_file(config_path, batch_settings, step_settings, 
                      model_name_or_path, # Added for reference
                      use_lora=False, use_qlora=False, quantization=None,
                      lora_rank=8, lora_alpha=16, lora_dropout=0.05,
                      precision="float16", gradient_checkpointing=True):
    
    backup_path = Path(str(config_path) + ".bak")
    try:
        shutil.copy2(config_path, backup_path)
        print(f"Backed up original config to: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not back up config file: {e}")

    # Using PyYAML for simplicity, ruamel.yaml can be added for comment preservation if needed
    with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Helper to ensure keys exist and set value
    def ensure_and_set(conf, path_list, value):
        curr = conf
        for i, key in enumerate(path_list[:-1]): # Iterate until the parent of the target key
            if key not in curr or not isinstance(curr[key], dict):
                curr[key] = {} # Create dict if not exists or not a dict
            curr = curr[key]
        curr[path_list[-1]] = value
        return conf
    
    # Model identifier (useful for tracking)
    ensure_and_set(config, ['model_args', 'model_name_or_path'], model_name_or_path)
    
    # Update batch settings
    ensure_and_set(config, ['training_args', 'per_device_train_batch_size'], batch_settings[0]) # Common HF Trainer arg
    ensure_and_set(config, ['training_args', 'gradient_accumulation_steps'], batch_settings[1])
    
    # Update step settings
    ensure_and_set(config, ['training_args', 'max_steps'], step_settings['total_steps'])
    ensure_and_set(config, ['training_args', 'warmup_steps'], step_settings['warmup_steps'])
    ensure_and_set(config, ['training_args', 'save_steps'], step_settings['save_steps'])
    ensure_and_set(config, ['training_args', 'eval_steps'], step_settings['eval_steps']) # Often evaluation_strategy='steps' is implied
    ensure_and_set(config, ['training_args', 'logging_steps'], step_settings['logging_steps'])
    
    # Precision
    ensure_and_set(config, ['training_args', 'fp16'], precision == "float16")
    ensure_and_set(config, ['training_args', 'bf16'], precision == "bfloat16")
    
    # Gradient checkpointing (often a model arg or trainer arg)
    ensure_and_set(config, ['training_args', 'gradient_checkpointing'], gradient_checkpointing)
    # Some frameworks put it in model_args:
    # ensure_and_set(config, ['model_args', 'gradient_checkpointing'], gradient_checkpointing)


    # PEFT / LoRA / QLoRA settings
    # Common structure for PEFT configs (e.g. Llama-Factory, Axolotl like)
    # This part might need heavy customization based on the target YAML schema
    if 'peft_config' not in config: config['peft_config'] = {} # General PEFT section
    
    if use_lora or use_qlora:
        config['peft_config']['peft_type'] = "LORA"
        config['peft_config']['r'] = lora_rank
        config['peft_config']['lora_alpha'] = lora_alpha
        config['peft_config']['lora_dropout'] = lora_dropout
        # Common target modules - user might need to verify for their specific model type
        config['peft_config']['target_modules'] = [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj", # For MLP layers in Llama-like
            # "embed_tokens", "lm_head" # Sometimes targeted for full LoRA
        ] 
        # task_type is crucial for PEFT
        ensure_and_set(config, ['peft_config', 'task_type'], "CAUSAL_LM")

        
        if use_qlora:
            # QLoRA implies LoRA, plus quantization settings for the base model
            ensure_and_set(config, ['quantization_config', 'load_in_4bit'], quantization == '4bit')
            ensure_and_set(config, ['quantization_config', 'load_in_8bit'], quantization == '8bit')
            if quantization == '4bit':
                ensure_and_set(config, ['quantization_config', 'bnb_4bit_compute_dtype'], precision) # torch.float16, torch.bfloat16
                ensure_and_set(config, ['quantization_config', 'bnb_4bit_quant_type'], "nf4") # common default
                ensure_and_set(config, ['quantization_config', 'bnb_4bit_use_double_quant'], True) # common default
            # Remove top-level load_in_4bit/8bit if they exist and we use quantization_config
            if 'load_in_4bit' in config: del config['load_in_4bit']
            if 'load_in_8bit' in config: del config['load_in_8bit']

    elif 'peft_config' in config: # Explicitly disable if PEFT was configured but not selected
        config['peft_config']['peft_type'] = "NONE" # Or remove the section

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"\nUpdated config file: {config_path}")
    print(f"  Model: {model_name_or_path}")
    print(f"  Precision: {precision}, Grad Checkpointing: {gradient_checkpointing}")
    eff_bs = batch_settings[0] * batch_settings[1]
    print(f"  Batch size: {batch_settings[0]}, Accumulation: {batch_settings[1]}, Effective Batch: {eff_bs}")
    print(f"  Total Steps: {step_settings['total_steps']}, Steps/Epoch: {step_settings['steps_per_epoch']}")
    print(f"  Warmup: {step_settings['warmup_steps']}, Save: {step_settings['save_steps']}, Eval: {step_settings['eval_steps']}, Log: {step_settings['logging_steps']}")
    if use_qlora:
        print(f"  Using QLoRA (quant: {quantization}, rank: {lora_rank}, alpha: {lora_alpha})")
    elif use_lora:
        print(f"  Using LoRA (rank: {lora_rank}, alpha: {lora_alpha})")
    
    print(f"\nReminder: If effective batch size ({eff_bs}) has changed significantly from a previous run,")
    print(f"  you might need to adjust the learning rate. Common heuristics involve scaling it")
    print(f"  linearly or with a square root rule relative to the original effective batch size and LR.")
        
def main():
    parser = argparse.ArgumentParser(
        description="Automatically configure training parameters in a YAML file based on GPU memory, Hugging Face model specs, and target GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model & Config
    model_group = parser.add_argument_group('Model & Configuration')
    model_group.add_argument("--model_name_or_path", type=str, required=True,
                             help="Name or path of the Hugging Face model (e.g., 'meta-llama/Llama-2-7b-hf').")
    model_group.add_argument("--config", type=str, required=True, 
                             help="Path to training config.yaml file to update.")
    model_group.add_argument("--dataset_size", type=int, required=True, 
                             help="Number of training examples in the dataset.")

    # Target Hardware
    hw_group = parser.add_argument_group('Target Hardware')
    hw_group.add_argument("--target_gpu_profile", type=str, default="local", choices=list(GPU_PROFILES.keys()),
                          help="Select a target GPU profile for VRAM estimation, or 'local' to detect." )
    
    # Training Hyperparameters
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    train_group.add_argument("--seq_length", type=int, default=2048, 
                             help="Maximum sequence length for model input.")
    default_precision = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    train_group.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default=default_precision,
                             help="Training precision.")
    train_group.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True,
                             help="Enable/disable gradient checkpointing.")
    
    # PEFT (LoRA/QLoRA) Parameters
    peft_group = parser.add_argument_group('PEFT (Parameter-Efficient Fine-Tuning)')
    peft_group.add_argument("--lora", action=argparse.BooleanOptionalAction, default=False, 
                            help="Enable LoRA fine-tuning.")
    peft_group.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=False, 
                            help="Enable QLoRA fine-tuning (implies LoRA with base model quantization).")
    peft_group.add_argument("--quantization", choices=["4bit", "8bit"], default="4bit", 
                            help="Base model quantization precision for QLoRA.")
    peft_group.add_argument("--lora_rank", type=int, default=32, help="LoRA rank (r).") # Increased default
    peft_group.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha (often 2*rank).") # Increased default
    peft_group.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability.")
    
    # System & Dataloader Parameters
    sys_group = parser.add_argument_group('System & Dataloader')
    sys_group.add_argument("--dataloader_workers", type=int, default=min(4, os.cpu_count() // 2 if os.cpu_count() else 1),
                           help="Number of dataloader workers for prefetching.")
    sys_group.add_argument("--system_reserve_gpu_gb", type=float, default=2.5, # Increased default for safety
                           help="GPU memory (GB) to reserve for OS, CUDA overhead, and non-PyTorch processes.")
    sys_group.add_argument("--dry-run", action="store_true", default=False,
                           help="Run calculations only, don't update the config file.")
    
    args = parser.parse_args()
    
    # --- Argument Validation and Processing ---
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return

    if args.qlora: args.lora = True # QLoRA implies LoRA is active for adapters

    # --- Fetch Model Architecture from Hugging Face ---
    print(f"Fetching model configuration for: {args.model_name_or_path}...")
    model_arch = get_hf_model_architecture(args.model_name_or_path)
    if not model_arch or model_arch.get("estimated_total_params_billions") is None:
        print(f"Could not fetch or reliably parse model architecture for {args.model_name_or_path}.")
        print("Please ensure the model name is correct and public, or that the config is standard.")
        # Provide an option here to fallback to manual parameter input if desired, or exit.
        # For now, we'll exit if critical info is missing.
        if not (model_arch and model_arch.get("hidden_size") and model_arch.get("num_hidden_layers")):
            print("Exiting due to missing critical model architecture details.")
            return 
        # If only params are missing, we might issue a warning but try to proceed if other arch details are present.
        if model_arch.get("estimated_total_params_billions") is None:
            print("Warning: Could not estimate total parameters. Memory calculations for weights/optimizer might be less accurate.")
            # Assign a placeholder if other details are fine, so it doesn't crash, but with a clear warning.
            # This part is tricky; robust fallback needs more logic or user input.
            # For now, if any key part is None after fetching, it will error out in calculate_batch_settings.

    # --- Get Target GPU Memory Info ---
    gpu_memory_info = get_gpu_memory_info(args.target_gpu_profile)
    available_gpu_for_calc = gpu_memory_info["available_for_new_pytorch_alloc"]
    
    print(f"Targeting GPU VRAM: {available_gpu_for_calc:.2f} GB ({args.target_gpu_profile})")
    if not gpu_memory_info["is_profiled"] and args.target_gpu_profile == "local":
         print(f"  Local GPU Info: Total={gpu_memory_info['total']:.2f}GB, PyTorch Reserved={gpu_memory_info['reserved_pytorch']:.2f}GB, Raw Driver Free={gpu_memory_info['raw_available']:.2f}GB")


    # --- System RAM Info ---
    ram_memory = get_ram_memory_info()
    print(f"System RAM Info: Total={ram_memory['total']:.2f}GB, Available={ram_memory['available']:.2f}GB")
    # Estimate RAM for dataloaders (using a placeholder effective batch size for now)
    # This is just for an informational print; not directly used to limit GPU batch size.
    placeholder_eff_batch = 64 
    dataset_ram_est = calculate_dataset_memory_ram(args.dataset_size, args.seq_length, placeholder_eff_batch, args.dataloader_workers)
    print(f"Estimated System RAM impact for dataset handling/prefetching (approx): {dataset_ram_est['ram_impact_gb']:.2f} GB")
    if dataset_ram_est['ram_impact_gb'] > ram_memory['available'] * 0.75: # If >75% of available
        print("Warning: Estimated dataset RAM impact is very high compared to available system RAM. System might become slow or OOM on CPU.")

    # --- Core Calculations ---
    batch_size, accumulation_steps = calculate_batch_settings(
        available_gpu_for_calc, model_arch,
        seq_length=args.seq_length,
        use_lora=args.lora, use_qlora=args.qlora, quantization=args.quantization, 
        lora_rank=args.lora_rank,
        precision=args.precision,
        system_reserve_gpu_gb=args.system_reserve_gpu_gb,
        gradient_checkpointing=args.gradient_checkpointing
    )
    
    step_settings = calculate_training_steps(
        args.dataset_size, batch_size, accumulation_steps, args.epochs
    )
    
    # --- Update Config File ---
    if step_settings["total_steps"] > 0:
        if args.dry_run:
            print("\nDRY RUN MODE: Skipping config file update.")
            print(f"  Config file '{config_path}' was NOT modified.")
            print("  The following settings would have been applied:")
            print(f"  - Model: {args.model_name_or_path}")
            print(f"  - Precision: {args.precision}, Grad Checkpointing: {args.gradient_checkpointing}")
            eff_bs = batch_size * accumulation_steps
            print(f"  - Batch size: {batch_size}, Accumulation: {accumulation_steps}, Effective Batch: {eff_bs}")
            print(f"  - Total Steps: {step_settings['total_steps']}, Steps/Epoch: {step_settings['steps_per_epoch']}")
            print(f"  - Warmup: {step_settings['warmup_steps']}, Save: {step_settings['save_steps']}, Eval: {step_settings['eval_steps']}, Log: {step_settings['logging_steps']}")
        else:
            update_config_file(
                config_path, 
                (batch_size, accumulation_steps), 
                step_settings,
                args.model_name_or_path,
                use_lora=args.lora, use_qlora=args.qlora, quantization=args.quantization,
                lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                precision=args.precision, gradient_checkpointing=args.gradient_checkpointing
            )
    else:
        print("\nSkipping configuration file update due to invalid step calculation (total_steps is 0).")
        print("This usually means the effective batch size is too large for the dataset size and number of epochs, or a calculation error occurred.")

if __name__ == "__main__":
    main()