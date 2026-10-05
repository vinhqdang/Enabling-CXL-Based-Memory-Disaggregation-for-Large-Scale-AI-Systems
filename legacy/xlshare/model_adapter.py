"""
HuggingFace Model Adapter for XL-Share system.
Handles conversion of PyTorch models to XL-Share native format.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from transformers import AutoModel, AutoConfig

from .inference_engine import ModelConfig
from .prefetcher import LayerInfo, LayerType

# Map torch modules to LayerType
MODULE_MAPPING = {
    nn.Linear: LayerType.LINEAR,
    nn.Conv2d: LayerType.CONV2D,
    nn.Embedding: LayerType.EMBEDDING,
    nn.LayerNorm: LayerType.NORMALIZATION,
    nn.MultiheadAttention: LayerType.ATTENTION,
    # Add more mappings as needed
}

class HFModelAdapter:
    """
    Adapter to convert HuggingFace models to XL-Share format.
    Extracts weights and computation graph structure.
    """
    
    @staticmethod
    def load_model(model_name: str, cache_dir: Optional[str] = None) -> Tuple[ModelConfig, Dict[str, np.ndarray]]:
        """
        Load and convert a HuggingFace model.
        
        Args:
            model_name: HF model identifier (e.g., 'bert-base-uncased')
            cache_dir: Optional directory for caching weights
            
        Returns:
            Tuple of (ModelConfig, weight_dict)
        """
        print(f"Loading model: {model_name}...")
        
        try:
            # Load model and config
            config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            model = AutoModel.from_pretrained(model_name, config=config, cache_dir=cache_dir)
            model.eval() # Set to evaluation mode
            
            layers: List[LayerInfo] = []
            weights: Dict[str, np.ndarray] = {}
            total_params = 0
            
            # Helper to process a module
            def process_module(name: str, module: nn.Module) -> Optional[LayerInfo]:
                # Skip container modules, only leaf modules with weights
                if len(list(module.children())) > 0:
                    return None
                    
                # Identify layer type
                layer_type = None
                for cls, ltype in MODULE_MAPPING.items():
                    if isinstance(module, cls):
                        layer_type = ltype
                        break
                
                # Check for transformer specific layers by name heuristics if type lookup fails
                if layer_type is None:
                    if "attention" in name.lower():
                        layer_type = LayerType.ATTENTION
                    elif "layer_norm" in name.lower() or "layernorm" in name.lower():
                        layer_type = LayerType.NORMALIZATION
                    elif "embedding" in name.lower():
                        layer_type = LayerType.EMBEDDING
                    elif "activation" in name.lower() or "act" in name.lower():
                         # Activations usually don't have weights, so might skip or mark as lightweight
                         return None
                    else:
                        # Fallback for unknown weighted layers (likely linear projections)
                        if hasattr(module, 'weight') and module.weight is not None:
                             layer_type = LayerType.LINEAR
                        else:
                             return None

                # Extract weights
                layer_weights = []
                if hasattr(module, 'weight') and module.weight is not None:
                    w = module.weight.detach().numpy()
                    layer_weights.append(w)
                if hasattr(module, 'bias') and module.bias is not None:
                    b = module.bias.detach().numpy()
                    layer_weights.append(b)
                
                if not layer_weights:
                    return None
                    
                # Combine weights into single array for storage simplified simulation
                # In a real system, we'd store W and b separate, but for XL-Share simulation
                # we treat "weights" as the data blob to fetch.
                # For computation, we will need to reconstitute them.
                # Here we will flatten and concatenate for the "blob", but for REAL execution
                # we need to preserve them.
                #
                # Strategy: 
                # Store the *primary* weight matrix as the blob for prefetching simulation.
                # For basic simulation, biases are small enough to be neglected or packed.
                # For Real Execution mode in Phase 1, we actually need the real objects.
                #
                # Let's simple store the state_dict for this module as a flat byte array 
                # for the "storage" simulation, but ALSO keep the structured weights 
                # if we want to run real inference.
                #
                # Wait, XL-Share is about *memory disaggregation*.
                # If we want to RUN real inference, we need to fetch the bytes,
                # deserialize them back to Tensor, and compute.
                
                primary_weight = layer_weights[0]
                
                # Computation cost estimation
                comp_time = 0.0
                if layer_type == LayerType.LINEAR:
                    comp_time = np.prod(primary_weight.shape) / 1e9 # GFLOPs estimate
                elif layer_type == LayerType.ATTENTION:
                    comp_time = np.prod(primary_weight.shape) / 0.5e9
                else:
                    comp_time = primary_weight.nbytes / 100e9 # Bandwidth bound
                
                # Reuse frequency heuristic
                reuse = 1
                if layer_type in [LayerType.EMBEDDING, LayerType.NORMALIZATION]:
                    reuse = 10
                
                info = LayerInfo(
                    name=name,
                    layer_type=layer_type,
                    weight_shape=primary_weight.shape,
                    weight_size_bytes=primary_weight.nbytes,
                    computation_time_ms=comp_time * 1000,
                    reuse_frequency=reuse
                )

                # Store only the primary weight as the blob for this layer.
                # Previously this concatenated weight+bias into one blob while
                # weight_size_bytes/weight_shape only accounted for the primary
                # weight -- the simulator's _deserialize_weights() then failed
                # to reshape the (larger) fetched blob back into weight_shape
                # for any layer with a bias (LayerNorm always has one; many
                # Linear/Conv1D layers do too). Matches this function's own
                # documented intent: biases are small enough to be neglected
                # for the transfer-size/prefetching simulation.
                weights[name] = primary_weight.flatten()
                
                return info

            # Flatten the model structure to a list of layers
            # We assume a sequential execution order match iteration order (mostly true for standard HF models)
            for name, module in model.named_modules():
                info = process_module(name, module)
                if info:
                    layers.append(info)
                    total_params += np.prod(info.weight_shape)

            total_size_mb = (total_params * 4) / (1024 * 1024)

            model_config = ModelConfig(
                name=model_name,
                layers=layers,
                total_params=total_params,
                total_size_mb=total_size_mb,
                architecture=config.model_type
            )
            
            print(f"Converted {model_name}: {len(layers)} layers, {total_size_mb:.2f}MB")
            return model_config, weights

        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

