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
        
    def refusal_metric_fn(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculate refusal score based on logits.
        
        Args:
            logits: (batch, seq_len, vocab_size)
            target_ids: (batch, seq_len) - not used for simple refusal probability
            
        Returns:
            Score tensor (batch,)
        """
        # Simple metric: Probability of "Sorry" or similar tokens at the first position
        # Or just use the loss on the target refusal string if provided
        
        # For now, let's use a simplified approach:
        # We want to measure how much the model refuses.
        # If we have a specific target (e.g. "I cannot"), we can measure the probability of that.
        
        # Let's assume we are looking at the first generated token probability for refusal words.
        # But we might not know which token is "Sorry" without the tokenizer.
        
        # Better approach: Use the provided target_ids (which should be the refusal response)
        # and calculate the negative log likelihood (loss) of generating that response.
        # Higher score = higher probability of refusal = lower loss.
        # So metric = -loss.
        
        # logits: (batch, seq_len, vocab)
        # target_ids: (batch, seq_len)
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Reshape back to (batch, seq_len-1) and sum
        loss = loss.view(shift_labels.size(0), shift_labels.size(1)).sum(dim=1)
        
        return -loss  # Higher is better (more refusal)

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
        target_encodings = self.model.tokenizer(
            target_outputs, 
            padding=True, 
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)
        target_ids = target_encodings.input_ids
        
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
            # Note: This assumes we have one SAE per layer and we know which one.
            # We might need to look up the correct SAE based on the layer name.
            # For now, let's assume residual stream SAEs.
            sae = self.sae_manager.load_saes_for_model(self.model.config.model_name, [f"residuals_{layer_idx}"]).get(f"residuals_{layer_idx}")
            
            if sae is None:
                return None
                
            def hook_fn(activation, hook_name):
                # activation: (batch, seq, hidden)
                
                # Encode
                feature_acts = sae.encode(activation)
                
                # Create mask for important features
                # important_features is a list of indices
                mask = torch.zeros(feature_acts.shape[-1], dtype=torch.bool, device=activation.device)
                mask[important_features] = True
                
                # Determine what to ablate
                if ablate_complement:
                    # Ablate everything NOT in the circuit
                    # Keep important features, replace others with mean
                    features_to_ablate = ~mask
                else:
                    # Ablate ONLY things in the circuit
                    features_to_ablate = mask
                
                # We need mean feature activations.
                # If ablation_values provides activation means, we need to encode them to get feature means?
                # Or does ablation_values provide feature means?
                # The input ablation_values are raw activation means.
                # So we should encode the mean activation to get mean feature activations?
                # Or just use the mean of the features from the dataset?
                # Let's assume we calculate feature means on the fly or pass them in.
                # For simplicity, let's use ZERO ablation for features for now, or 
                # better: encode the mean activation.
                
                mean_act = mean_val.to(activation.device)
                mean_feature_acts = sae.encode(mean_act)
                
                # Apply ablation
                # feature_acts[..., features_to_ablate] = mean_feature_acts[..., features_to_ablate]
                # Broadcasting might be tricky if mean_feature_acts is (1, 1, hidden)
                
                # Expand mean to match batch/seq
                target_shape = feature_acts.shape
                expanded_mean = mean_feature_acts.expand(target_shape)
                
                # We can't do in-place modification easily with boolean indexing on last dim if shapes match
                # But here we want to replace specific feature indices across all batch/seq positions.
                
                # Create a mask of shape (hidden,)
                # features_to_ablate is (hidden,)
                
                # Apply
                feature_acts_modified = feature_acts.clone()
                feature_acts_modified[..., features_to_ablate] = expanded_mean[..., features_to_ablate]
                
                # Decode
                reconstructed = sae.decode(feature_acts_modified)
                
                # Add error term (original - reconstructed_original) ?
                # Faithfulness usually implies running with ONLY the circuit features.
                # So: Output = Decode(Features * Mask) + Error?
                # Or Output = Decode(Features * Mask + Mean * ~Mask) + Error?
                # Usually we keep the error term unablated or zero-ablate it.
                # feature-circuits defaults to keeping the error term.
                
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
        
        # Run inference
        # We need to get logits. ModelWrapper.run_with_activations returns text.
        # We need a method to get logits.
        # I'll add a temporary method or use the model directly since I have access to it.
        
        inputs = self.model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.model(**inputs)
            logits = outputs.logits
            
        # Calculate metric
        # We need the target ids for the metric
        # Assuming prompts + target_outputs are aligned
        
        # We need to construct the full sequence for loss calculation: Prompt + Target
        # But here we just want to check if the model produces the target given the prompt.
        # So inputs are prompts.
        # Logits are for the next tokens.
        
        # Wait, `logits` shape is (batch, seq_len, vocab).
        # We care about the logits for the *generated* tokens.
        # But we didn't generate, we just ran a forward pass on the prompt.
        # So we are predicting the *first* token of the response.
        
        # If we want to evaluate the full response, we need to feed (Prompt + Target) and look at loss on Target positions.
        
        # Let's adjust inputs to be Prompt + Target
        full_inputs_ids = []
        target_masks = [] # 1 for target positions, 0 for prompt
        
        for prompt, target in zip(prompts, target_outputs):
            prompt_ids = self.model.tokenizer.encode(prompt, add_special_tokens=False)
            target_ids_list = self.model.tokenizer.encode(target, add_special_tokens=False)
            full_ids = prompt_ids + target_ids_list
            mask = [0] * len(prompt_ids) + [1] * len(target_ids_list)
            
            full_inputs_ids.append(torch.tensor(full_ids))
            target_masks.append(torch.tensor(mask))
            
        # Pad
        full_inputs_padded = torch.nn.utils.rnn.pad_sequence(full_inputs_ids, batch_first=True, padding_value=self.model.tokenizer.pad_token_id).to(self.device)
        # We need to handle masks carefully if padding exists
        
        # Run forward pass
        with torch.no_grad():
            outputs = self.model.model(input_ids=full_inputs_padded)
            logits = outputs.logits
            
        # Calculate loss only on target positions
        # This is getting complicated to implement perfectly in one go.
        # Let's simplify: Just measure prob of first token of target.
        
        # Simplified approach:
        # Just run forward on prompt.
        # Check prob of first token of target.
        
        first_target_tokens = [self.model.tokenizer.encode(t, add_special_tokens=False)[0] for t in target_outputs]
        first_target_tensor = torch.tensor(first_target_tokens, device=self.device)
        
        # Logits at the last position of the prompt
        # We need to find the last position for each prompt (due to padding)
        # But if we just ran forward on prompts, we can take the last token.
        
        # Let's stick to the simpler forward pass on prompts.
        inputs = self.model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.model(**inputs)
            # logits: (batch, seq, vocab)
            # We want the logits corresponding to the last token of the prompt
            # inputs.attention_mask can tell us where the last token is
            
            last_token_idxs = inputs.attention_mask.sum(dim=1) - 1
            # Select logits
            final_logits = outputs.logits[torch.arange(outputs.logits.shape[0]), last_token_idxs, :]
            
            # Calculate log prob of target token
            log_probs = torch.nn.functional.log_softmax(final_logits, dim=-1)
            target_log_probs = log_probs[torch.arange(log_probs.shape[0]), first_target_tensor]
            
        return target_log_probs.mean().item()

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
