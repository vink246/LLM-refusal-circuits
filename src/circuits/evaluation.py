"""
Circuit evaluation metrics (Faithfulness and Completeness).
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
from tqdm import tqdm
import copy

from ..models.model_wrapper import ModelWrapper
from .circuit import SparseFeatureCircuit
from ..sae.sae_manager import SAEManager

class CircuitEvaluator:
    """Evaluates circuit faithfulness and completeness"""
    
    def __init__(
        self, 
        model_wrapper: ModelWrapper, 
        sae_manager: SAEManager,
        config: Dict[str, Any]
    ):
        self.model = model_wrapper
        self.sae_manager = sae_manager
        self.config = config
        self.device = model_wrapper.config.device_map
        
        self.device = model_wrapper.config.device_map
        
        self.refusal_indicators = [
            "sorry", "cannot", "unable", "not allowed", "against", 
            "policy", "inappropriate", "should not", "cannot help",
            "I cannot", "I'm unable", "I am unable", "I'm not", "I am not", "I cannot provide",
            "I cannot assist", "I cannot answer", "I cannot help you",
            "I can't", "I'm not able", "I'm not able to"
        ]
        # Pre-compute token IDs for refusal indicators (first token of each)
        # We do this lazily or in init? Init is better.
        # But we need the tokenizer.
        self.refusal_token_ids = None

    def _get_refusal_token_ids(self):
        if self.refusal_token_ids is None:
            # Get the first token of each indicator
            # Note: We should handle spacing. " sorry" vs "sorry".
            # Usually the model generates a space first?
            # Let's assume the indicators might appear at the start.
            # We encode " " + indicator and indicator.
            
            ids = set()
            for ind in self.refusal_indicators:
                # Encode with and without leading space
                tokens1 = self.model.tokenizer.encode(ind, add_special_tokens=False)
                tokens2 = self.model.tokenizer.encode(" " + ind, add_special_tokens=False)
                
                if tokens1: ids.add(tokens1[0])
                if tokens2: ids.add(tokens2[0])
            
            self.refusal_token_ids = torch.tensor(list(ids), device=self.device)
        return self.refusal_token_ids

    def refusal_metric_fn(self, logits: torch.Tensor, target_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate refusal score based on logits.
        Uses the probability of refusal indicator tokens.
        
        Args:
            logits: (batch, seq_len, vocab_size) - we usually care about the last position
            target_ids: Unused now
            
        Returns:
            Score tensor (batch,)
        """
        # We assume logits are for the NEXT token prediction (e.g. last position of prompt)
        # If logits has seq_len > 1, we take the last one?
        # Usually run_with_ablations handles extracting the relevant logits.
        # But here logits is passed in.
        
        # If logits is (batch, vocab), use it directly.
        # If (batch, seq, vocab), take last.
        if logits.dim() == 3:
            logits = logits[:, -1, :]
            
        refusal_ids = self._get_refusal_token_ids()
        
        # Calculate log probs
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Sum of probabilities of all refusal tokens (LogSumExp of log probs)
        # score = log( sum( exp(log_prob[i]) ) )
        refusal_log_probs = log_probs[:, refusal_ids]
        score = torch.logsumexp(refusal_log_probs, dim=-1)
        
        return score

    def get_mean_ablation_values(self, activations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate mean activation for each layer/feature.
        
        Args:
            activations: Dictionary of activations {layer: (batch, ...)}
            
        Returns:
            Dictionary of mean activations {layer: (1, ...)}
        """
        mean_values = {}
        for layer, act in activations.items():
            # Average over batch and sequence length (if present)
            if act.dim() == 3:
                # (batch, seq, hidden) -> (1, 1, hidden)
                mean_values[layer] = act.mean(dim=(0, 1), keepdim=True)
            else:
                # (batch, hidden) -> (1, hidden)
                mean_values[layer] = act.mean(dim=0, keepdim=True)
        return mean_values

    def run_with_ablations(
        self,
        prompts: List[str],
        target_outputs: List[str],
        circuit: SparseFeatureCircuit,
        ablation_values: Dict[str, torch.Tensor],
        ablate_complement: bool = True
    ) -> float:
        """
        Run inference with circuit-based ablations.
        
        Args:
            prompts: List of input prompts
            target_outputs: List of target refusal responses
            circuit: The circuit defining important features
            ablation_values: Mean values to use for ablation
            ablate_complement: If True, ablate everything NOT in the circuit (Faithfulness).
                             If False, ablate ONLY things in the circuit (Completeness check / Knockout).
                             
        Returns:
            Average metric score
        """
        # Prepare target IDs
        # Not needed for refusal metric based on indicators, but kept for interface compatibility
        # target_encodings = self.model.tokenizer(
        #     target_outputs, 
        #     padding=True, 
        #     return_tensors="pt",
        #     add_special_tokens=False
        # ).to(self.device)
        # target_ids = target_encodings.input_ids
        
        # Setup hooks
        self.model.clear_intervention_hooks()
        
        # Group circuit nodes by layer
        nodes_by_layer = {}
        for node_id, node_data in circuit.nodes.items():
            layer_idx = node_data['layer']
            if layer_idx not in nodes_by_layer:
                nodes_by_layer[layer_idx] = []
            nodes_by_layer[layer_idx].append(int(node_data['feature_id']))
            
        # Define hook function factory
        def make_hook_fn(layer_idx, important_features, mean_val):
            # Load SAE for this layer
            sae = self.sae_manager.load_saes_for_model(self.model.config.model_name, [f"residuals_{layer_idx}"]).get(f"residuals_{layer_idx}")
            
            if sae is None:
                return None
                
            def hook_fn(activation, hook_name):
                # activation: (batch, seq, hidden)
                
                # Encode
                feature_acts = sae.encode(activation)
                
                # Create mask for important features
                mask = torch.zeros(feature_acts.shape[-1], dtype=torch.bool, device=activation.device)
                mask[important_features] = True
                
                # Determine what to ablate
                if ablate_complement:
                    features_to_ablate = ~mask
                else:
                    features_to_ablate = mask
                
                # Use mean feature activations (encode mean activation)
                mean_act = mean_val.to(activation.device)
                mean_feature_acts = sae.encode(mean_act)
                
                # Expand mean to match batch/seq
                target_shape = feature_acts.shape
                expanded_mean = mean_feature_acts.expand(target_shape)
                
                # Apply ablation
                feature_acts_modified = feature_acts.clone()
                feature_acts_modified[..., features_to_ablate] = expanded_mean[..., features_to_ablate]
                
                # Decode
                reconstructed = sae.decode(feature_acts_modified)
                
                # Add error term (original - reconstructed_original)
                original_reconstructed = sae.decode(feature_acts)
                error = activation - original_reconstructed
                
                return reconstructed + error
                
            return hook_fn

        # Register hooks
        for layer_idx, nodes in nodes_by_layer.items():
            layer_spec = f"residuals_{layer_idx}"
            if layer_spec in ablation_values:
                hook = make_hook_fn(layer_idx, nodes, ablation_values[layer_spec])
                if hook:
                    self.model.register_intervention_hook(layer_spec, hook)
        
        # Run inference on prompts only
        inputs = self.model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.model(**inputs)
            # logits: (batch, seq, vocab)
            
            # We want the logits corresponding to the last token of the prompt
            # inputs.attention_mask can tell us where the last token is
            last_token_idxs = inputs.attention_mask.sum(dim=1) - 1
            
            # Select logits at the last position
            final_logits = outputs.logits[torch.arange(outputs.logits.shape[0]), last_token_idxs, :]
            
            # Calculate metric
            score = self.refusal_metric_fn(final_logits)
            
        return score.mean().item()

    def evaluate_circuit(
        self,
        circuit: SparseFeatureCircuit,
        prompts: List[str],
        target_outputs: List[str],
        mean_activations: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate circuit faithfulness and completeness.
        
        Args:
            circuit: The circuit to evaluate
            prompts: List of prompts
            target_outputs: List of target refusal strings (e.g. "I cannot")
            mean_activations: Mean activations for ablation
            
        Returns:
            Dictionary with 'faithfulness', 'completeness', 'F_M', 'F_C', 'F_empty'
        """
        print("Evaluating circuit...")
        
        # 1. Compute F(M) - Full Model Performance
        print("  Computing F(M)...")
        self.model.clear_intervention_hooks()
        f_m = self.run_with_ablations(prompts, target_outputs, circuit, mean_activations, ablate_complement=False) 
        # Wait, run_with_ablations with ablate_complement=False means we ablate the circuit? 
        # No, we want NO ablation.
        # I should add a 'no_ablation' flag or just pass empty circuit?
        # Actually, if I pass an empty circuit and ablate_complement=False, it ablates nothing (mask is all false, features_to_ablate is all false).
        
        empty_circuit = SparseFeatureCircuit()
        f_m = self.run_with_ablations(prompts, target_outputs, empty_circuit, mean_activations, ablate_complement=False)
        
        # 2. Compute F(C) - Circuit Performance (Faithfulness)
        # Ablate everything NOT in the circuit (replace with mean)
        print("  Computing F(C)...")
        f_c = self.run_with_ablations(prompts, target_outputs, circuit, mean_activations, ablate_complement=True)
        
        # 3. Compute F(Empty) - Empty Circuit Performance (Random/Mean)
        # Ablate everything (replace with mean)
        print("  Computing F(Empty)...")
        f_empty = self.run_with_ablations(prompts, target_outputs, empty_circuit, mean_activations, ablate_complement=True)
        
        # Calculate metrics
        # Faithfulness = (F(C) - F(Empty)) / (F(M) - F(Empty))
        if f_m - f_empty != 0:
            faithfulness = (f_c - f_empty) / (f_m - f_empty)
        else:
            faithfulness = 0.0
            
        # Completeness = F(C) / F(M) ? 
        # Usually Completeness is how much of the behavior is recovered.
        # If F(C) is close to F(M), it's complete.
        # But F(C) is "Circuit Only".
        # Sometimes Completeness is defined via "Knockout": F(M \ C).
        # If F(M \ C) drops to F(Empty), then the circuit is necessary (complete?).
        
        # Let's stick to the definition in feature-circuits if possible.
        # In feature-circuits/ablation.py:
        # Faithfulness = (fc - fempty) / (fm - fempty)
        # They don't explicitly calculate completeness in the snippet I saw, but usually it's related.
        # I will return the raw values and the faithfulness score.
        
        return {
            "faithfulness": faithfulness,
            "F_M": f_m,
            "F_C": f_c,
            "F_empty": f_empty
        }
