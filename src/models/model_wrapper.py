"""
Model wrapper for loading models and collecting activations during inference.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ModelConfig:
    """Configuration for model loading"""
    model_name: str
    torch_dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = False


class ModelWrapper:
    """Wrapper for loading models and collecting activations via hooks"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.hooks = []
        self.activations = defaultdict(dict)
        
    def load_model(self):
        """Load the model and tokenizer"""
        print(f"Loading model: {self.config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=getattr(torch, self.config.torch_dtype) if hasattr(torch, self.config.torch_dtype) else torch.float16,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"✓ Model loaded on device: {next(self.model.parameters()).device}")
        
    def setup_activation_hooks(self, layers: List[str]):
        """
        Set up hooks for collecting activations from specified layers.
        
        Supports OR-Bench style layer specifications:
        - residuals_N: Residual stream at layer N
        - mlp_N: MLP output at layer N
        - attention_N: Attention output at layer N
        """
        self.activations.clear()
        self._remove_hooks()
        
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            raise ValueError(f"Model architecture not supported for hooking. Expected model.model.layers structure.")
        
        num_layers = len(self.model.model.layers)
        print(f"Setting up activation hooks for {len(layers)} layer(s) (model has {num_layers} layers)")
        
        for layer_spec in layers:
            if layer_spec.startswith('residuals_'):
                layer_idx = int(layer_spec.split('_')[1])
                if layer_idx >= num_layers:
                    print(f"Warning: Layer {layer_idx} >= {num_layers}, skipping")
                    continue
                self._hook_residual_layer(layer_idx)
            elif layer_spec.startswith('mlp_'):
                layer_idx = int(layer_spec.split('_')[1])
                if layer_idx >= num_layers:
                    print(f"Warning: Layer {layer_idx} >= {num_layers}, skipping")
                    continue
                self._hook_mlp_layer(layer_idx)
            elif layer_spec.startswith('attention_'):
                layer_idx = int(layer_spec.split('_')[1])
                if layer_idx >= num_layers:
                    print(f"Warning: Layer {layer_idx} >= {num_layers}, skipping")
                    continue
                self._hook_attention_layer(layer_idx)
            else:
                print(f"Warning: Unknown layer specification: {layer_spec}")
    
    def _hook_residual_layer(self, layer_idx: int):
        """Hook residual stream at specified layer"""
        layer_module = self.model.model.layers[layer_idx]
        
        def make_residual_hook(l_idx):
            def residual_hook(module, input, output):
                if len(input) > 0:
                    # Store activation (detach and move to CPU to save memory)
                    self.activations[f'residuals_{l_idx}'] = input[0].detach().cpu()
            return residual_hook
        
        self.hooks.append(layer_module.register_forward_hook(make_residual_hook(layer_idx)))
    
    def _hook_mlp_layer(self, layer_idx: int):
        """Hook MLP output at specified layer"""
        mlp_layer = self.model.model.layers[layer_idx].mlp
        
        def make_mlp_hook(l_idx):
            def mlp_hook(module, input, output):
                self.activations[f'mlp_{l_idx}'] = output.detach().cpu()
            return mlp_hook
        
        self.hooks.append(mlp_layer.register_forward_hook(make_mlp_hook(layer_idx)))
    
    def _hook_attention_layer(self, layer_idx: int):
        """Hook attention output at specified layer"""
        attention_layer = self.model.model.layers[layer_idx].self_attn
        
        def make_attention_hook(l_idx):
            def attention_hook(module, input, output):
                # Attention output is typically a tuple, we want the first element
                if isinstance(output, tuple):
                    self.activations[f'attention_{l_idx}'] = output[0].detach().cpu()
                else:
                    self.activations[f'attention_{l_idx}'] = output.detach().cpu()
            return attention_hook
        
        self.hooks.append(attention_layer.register_forward_hook(make_attention_hook(layer_idx)))
    
    def run_with_activations(
        self,
        prompts: List[str],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> tuple:
        """
        Run batch inference and return outputs with activations.
        
        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (if do_sample=True)
            do_sample: Whether to use sampling
            
        Returns:
            Tuple of (generated_texts, activations_dict)
        """
        self.activations.clear()
        
        # Tokenize with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Reasonable max length
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=1,
                output_hidden_states=False,  # We use hooks instead
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode generated texts
        generated_texts = []
        for i in range(len(prompts)):
            input_length = inputs['input_ids'].shape[1]
            generated_seq = outputs.sequences[i][input_length:]
            generated_text = self.tokenizer.decode(generated_seq, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts, dict(self.activations)
    
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def cleanup(self):
        """Clean up resources"""
        self._remove_hooks()
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def load_model_with_hooks(
    model_name: str,
    layers: List[str],
    device: str = "cuda",
    torch_dtype: str = "float16",
    trust_remote_code: bool = False
) -> ModelWrapper:
    """
    Factory function to load model with activation hooks.
    
    Args:
        model_name: HuggingFace model name
        layers: List of layer specifications (e.g., ['residuals_0', 'mlp_5'])
        device: Device to load model on
        torch_dtype: Torch dtype for model weights
        trust_remote_code: Whether to trust remote code
        
    Returns:
        ModelWrapper instance with hooks set up
    """
    config = ModelConfig(
        model_name=model_name,
        device_map=device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code
    )
    model_wrapper = ModelWrapper(config)
    model_wrapper.load_model()
    model_wrapper.setup_activation_hooks(layers)
    return model_wrapper

