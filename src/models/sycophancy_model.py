"""
SycophancyModel - TransformerLens wrapper for mechanistic interpretability analysis.

This class provides a unified interface for:
- Loading Llama-3, Mistral, or other HuggingFace models via TransformerLens
- Extracting logits and activations for sycophancy analysis
- Running causal interventions (activation patching, steering)
"""

import torch
from typing import List, Dict, Optional, Union
from transformer_lens import HookedTransformer
import warnings


class SycophancyModel:
    """
    Wrapper around TransformerLens HookedTransformer for sycophancy research.
    
    Usage:
        model = SycophancyModel("meta-llama/Meta-Llama-3-8B-Instruct")
        logits = model.get_logits(["Hello, how are you?"])
        activations = model.get_activations(["Hello"], layers=[10, 15, 20])
    """
    
    def __init__(
        self, 
        model_name: str,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        n_devices: int = 1,
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
            device: Device to load model on. If None, auto-detects GPU/CPU.
            dtype: Model dtype (float16 recommended for large models)
            n_devices: Number of devices for model parallelism (for multi-GPU setups)
        """
        self.model_name = model_name
        self.dtype = dtype
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                warnings.warn(
                    "No GPU detected. Loading on CPU will be slow for large models. "
                    "Consider using a smaller model like 'gpt2' for testing."
                )
        
        self.device = device
        
        print(f"Loading {model_name} on {device} with dtype={dtype}...")
        
        # Load via TransformerLens
        # TransformerLens will automatically download and convert HF models
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=dtype,
            n_devices=n_devices,
            fold_ln=False,  # Keep layer norms separate for interpretability
            center_writing_weights=False,
            center_unembed=False,
        )
        
        self.model.eval()
        self.n_layers = self.model.cfg.n_layers
        self.n_heads = self.model.cfg.n_heads
        self.d_model = self.model.cfg.d_model
        
        print(f"Loaded! Layers: {self.n_layers}, Heads: {self.n_heads}, d_model: {self.d_model}")
    
    @torch.no_grad()
    def get_logits(
        self, 
        prompts: Union[str, List[str]],
        return_type: str = "logits"
    ) -> torch.Tensor:
        """
        Get the model's output logits for given prompts.
        
        Args:
            prompts: Single prompt string or list of prompts
            return_type: What to return ("logits" or "loss")
            
        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size)
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        return self.model(prompts, return_type=return_type)
    
    @torch.no_grad()
    def get_activations(
        self,
        prompts: Union[str, List[str]],
        layers: Optional[List[int]] = None,
        components: List[str] = ["resid_post"],
    ) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate activations from specified layers.
        
        Args:
            prompts: Input prompts
            layers: Which layers to extract from (default: all)
            components: Which components to cache. Options:
                - "resid_pre": Residual stream before attention
                - "resid_post": Residual stream after MLP (main one for probing)
                - "attn_out": Attention output
                - "mlp_out": MLP output
                
        Returns:
            Dictionary mapping "component_layer" to activation tensors
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if layers is None:
            layers = list(range(self.n_layers))
        
        # Build the list of hook names
        hook_names = []
        for layer in layers:
            for comp in components:
                hook_names.append(f"blocks.{layer}.hook_{comp}")
        
        # Run with caching
        _, cache = self.model.run_with_cache(
            prompts,
            names_filter=lambda name: name in hook_names
        )
        
        return dict(cache)
    
    @torch.no_grad()
    def get_attention_patterns(
        self,
        prompts: Union[str, List[str]],
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract attention patterns from all heads at specified layers.
        
        Useful for identifying "sycophancy heads" that attend to user opinion tokens.
        
        Args:
            prompts: Input prompts
            layers: Which layers to analyze (default: all)
            
        Returns:
            Dictionary mapping layer index to attention pattern tensor
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if layers is None:
            layers = list(range(self.n_layers))
        
        hook_names = [f"blocks.{l}.attn.hook_pattern" for l in layers]
        
        _, cache = self.model.run_with_cache(
            prompts,
            names_filter=lambda name: name in hook_names
        )
        
        return {l: cache[f"blocks.{l}.attn.hook_pattern"] for l in layers}
    
    def evaluate_sycophancy(
        self,
        dataset: List[Dict],
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate sycophancy rate on a dataset.
        
        Args:
            dataset: List of dicts with 'biased_prompt', 'sycophantic_target', 'non_sycophantic_target'
            max_samples: Limit evaluation to N samples
            verbose: Print progress
            
        Returns:
            Dictionary with 'sycophancy_rate' and other metrics
        """
        from tqdm import tqdm
        
        if max_samples:
            dataset = dataset[:max_samples]
        
        sycophantic_count = 0
        total = 0
        
        iterator = tqdm(dataset, desc="Evaluating sycophancy") if verbose else dataset
        
        for item in iterator:
            prompt = item['biased_prompt']
            
            # Get logits for the prompt
            logits = self.get_logits(prompt)
            last_token_logits = logits[0, -1, :]
            
            # Get token IDs for targets
            try:
                syc_id = self.model.to_single_token(item['sycophantic_target'])
                non_syc_id = self.model.to_single_token(item['non_sycophantic_target'])
            except Exception:
                # If tokenization fails (multi-token), skip
                continue
            
            score_syc = last_token_logits[syc_id].item()
            score_non = last_token_logits[non_syc_id].item()
            
            if score_syc > score_non:
                sycophantic_count += 1
            
            total += 1
        
        rate = sycophantic_count / total if total > 0 else 0.0
        
        return {
            "sycophancy_rate": rate,
            "sycophantic_count": sycophantic_count,
            "total_evaluated": total,
        }
    
    def compute_steering_vector(
        self,
        neutral_prompts: List[str],
        sycophantic_prompts: List[str],
        layer: int,
        position: int = -1,
    ) -> torch.Tensor:
        """
        Compute a steering vector to reduce sycophancy.
        
        The steering vector is the mean difference between sycophantic and neutral activations.
        Subtracting this vector during inference can reduce sycophantic behavior.
        
        Args:
            neutral_prompts: Prompts without user opinion bias
            sycophantic_prompts: Prompts with user opinion bias
            layer: Which layer to compute the vector from
            position: Token position (-1 for last token)
            
        Returns:
            Steering vector of shape (d_model,)
        """
        # Get activations for both prompt sets
        neutral_acts = self.get_activations(neutral_prompts, layers=[layer])
        syc_acts = self.get_activations(sycophantic_prompts, layers=[layer])
        
        key = f"blocks.{layer}.hook_resid_post"
        
        # Extract at the specified position
        neutral_vec = neutral_acts[key][:, position, :].mean(dim=0)
        syc_vec = syc_acts[key][:, position, :].mean(dim=0)
        
        # Steering vector: sycophantic direction
        steering_vector = syc_vec - neutral_vec
        
        return steering_vector
    
    def generate_with_steering(
        self,
        prompt: str,
        steering_vector: torch.Tensor,
        layer: int,
        alpha: float = 1.0,
        max_new_tokens: int = 50,
    ) -> str:
        """
        Generate text while subtracting a steering vector to reduce sycophancy.
        
        Args:
            prompt: Input prompt
            steering_vector: Vector to subtract from activations
            layer: Which layer to apply steering
            alpha: Scaling factor for the steering vector
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        
        def steering_hook(activation, hook):
            # Subtract the steering vector (scaled by alpha) from all positions
            return activation - alpha * steering_vector.to(activation.device)
        
        hook_name = f"blocks.{layer}.hook_resid_post"
        
        # Generate with the hook active
        with self.model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
            output = self.model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
            )
        
        return output
    
    def __repr__(self) -> str:
        return f"SycophancyModel({self.model_name}, device={self.device})"

