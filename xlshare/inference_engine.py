"""
XL-Share Inference Engine

Main inference system that coordinates memory management, prefetching,
and model execution for CXL-based memory disaggregation.
"""

import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass

from .memory_manager import CXLMemoryManager, LocalCache
from .prefetcher import ModelAwarePrefetcher, LayerInfo, LayerType
from .emulator import CXLEmulator


@dataclass
class ModelConfig:
    """Configuration for a neural network model"""
    name: str
    layers: List[LayerInfo]
    total_params: int
    total_size_mb: float
    architecture: str = "transformer"


@dataclass
class InferenceRequest:
    """Represents an inference request"""
    request_id: str
    input_data: np.ndarray
    model_name: str
    timestamp: float
    priority: int = 0


@dataclass  
class InferenceResult:
    """Result of an inference request"""
    request_id: str
    output_data: np.ndarray
    latency_ms: float
    throughput_tokens_per_sec: float
    cache_hit_rate: float
    memory_stats: Dict[str, Any]


class XLShareInferenceEngine:
    """
    Main inference engine that orchestrates model execution using
    CXL-based memory disaggregation with intelligent prefetching.
    """
    
    def __init__(self, 
                 cxl_pool_size_gb: float = 64.0,
                 gpu_cache_size_mb: int = 8192,
                 emulate_cxl: bool = True):
        """
        Initialize XL-Share inference engine
        
        Args:
            cxl_pool_size_gb: Size of CXL memory pool in GB
            gpu_cache_size_mb: Size of local GPU cache in MB
            emulate_cxl: Whether to use CXL emulator or real hardware
        """
        # Initialize memory subsystem
        if emulate_cxl:
            self.cxl_emulator = CXLEmulator()
            # Use emulator for memory operations
            self.memory_manager = CXLMemoryManager(cxl_pool_size_gb)
        else:
            self.memory_manager = CXLMemoryManager(cxl_pool_size_gb)
            self.cxl_emulator = None
        
        self.local_cache = LocalCache(gpu_cache_size_mb)
        self.prefetcher = ModelAwarePrefetcher(self.memory_manager, self.local_cache)
        
        # Model registry
        self.models: Dict[str, ModelConfig] = {}
        self.model_addresses: Dict[str, Dict[str, int]] = {}
        
        # Execution statistics
        self.stats = {
            'total_requests': 0,
            'total_latency_ms': 0.0,
            'total_throughput': 0.0,
            'gpu_utilization': 0.0,
            'memory_efficiency': 0.0
        }
        
        self.execution_lock = threading.Lock()
        
        print(f"XL-Share Inference Engine initialized")
        print(f"  - CXL Pool: {cxl_pool_size_gb}GB")
        print(f"  - GPU Cache: {gpu_cache_size_mb}MB")
        print(f"  - Emulation: {emulate_cxl}")
    
    def register_model(self, model_config: ModelConfig, weights: Dict[str, np.ndarray]) -> bool:
        """
        Register a model for inference
        
        Args:
            model_config: Model configuration
            weights: Model weights dictionary
            
        Returns:
            True if registration successful
        """
        try:
            # Store weights in CXL memory
            weight_addresses = self.memory_manager.store_model_weights(weights)
            
            # Register with prefetcher
            self.prefetcher.register_model(model_config.layers, weight_addresses)
            
            # Store model info
            self.models[model_config.name] = model_config
            self.model_addresses[model_config.name] = weight_addresses
            
            print(f"Registered model '{model_config.name}':")
            print(f"  - Layers: {len(model_config.layers)}")
            print(f"  - Parameters: {model_config.total_params:,}")
            print(f"  - Size: {model_config.total_size_mb:.1f}MB")
            
            return True
            
        except Exception as e:
            print(f"Failed to register model: {e}")
            return False
    
    def inference(self, request: InferenceRequest) -> InferenceResult:
        """
        Execute inference request using XL-Share system
        
        Args:
            request: Inference request to process
            
        Returns:
            Inference result with performance metrics
        """
        start_time = time.time()
        
        if request.model_name not in self.models:
            raise ValueError(f"Model '{request.model_name}' not registered")
        
        with self.execution_lock:
            model_config = self.models[request.model_name]
            
            # Execute model layers with prefetching
            output = self._execute_model(
                model_config, 
                request.input_data,
                request.request_id
            )
            
            # Calculate performance metrics
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Estimate throughput (tokens/sec for language models)
            output_tokens = np.prod(output.shape)
            throughput = output_tokens / (latency_ms / 1000)
            
            # Get cache statistics
            cache_stats = self.local_cache.get_stats()
            prefetch_stats = self.prefetcher.get_stats()
            memory_stats = self.memory_manager.get_stats()
            
            # Update global statistics
            self.stats['total_requests'] += 1
            self.stats['total_latency_ms'] += latency_ms
            self.stats['total_throughput'] += throughput
            
            result = InferenceResult(
                request_id=request.request_id,
                output_data=output,
                latency_ms=latency_ms,
                throughput_tokens_per_sec=throughput,
                cache_hit_rate=cache_stats['hit_rate'],
                memory_stats={
                    'cache': cache_stats,
                    'prefetch': prefetch_stats,
                    'memory': memory_stats
                }
            )
            
            return result
    
    def _execute_model(self, model_config: ModelConfig, input_data: np.ndarray, 
                      request_id: str) -> np.ndarray:
        """
        Execute model layers with intelligent prefetching
        
        Args:
            model_config: Model configuration
            input_data: Input tensor
            request_id: Request identifier
            
        Returns:
            Model output tensor
        """
        current_input = input_data
        execution_order = self.prefetcher.execution_order
        
        print(f"Executing model '{model_config.name}' (request: {request_id})")
        
        for layer_idx, layer_name in enumerate(execution_order):
            layer_info = self.prefetcher.layers[layer_name]
            
            # Start prefetching for upcoming layers
            self.prefetcher.smart_prefetch_pipeline(layer_idx, lookahead=2)
            
            # Wait for current layer weights
            layer_start_time = time.time()
            weights = self.prefetcher.wait_for_weights(layer_name)
            weight_load_time = time.time() - layer_start_time
            
            # Execute layer computation
            compute_start_time = time.time()
            current_input = self._execute_layer(layer_info, current_input, weights)
            compute_time = time.time() - compute_start_time
            
            # Mark weights for eviction if not frequently reused
            if layer_info.reuse_frequency <= 1:
                self.local_cache.mark_for_eviction(layer_name)
            
            # Log layer execution
            print(f"  {layer_name}: weight_load={weight_load_time*1000:.1f}ms, "
                  f"compute={compute_time*1000:.1f}ms")
        
        return current_input
    
    def _execute_layer(self, layer_info: LayerInfo, input_data: np.ndarray, 
                      weights: np.ndarray) -> np.ndarray:
        """
        Execute a single neural network layer
        
        Args:
            layer_info: Layer information
            input_data: Layer input tensor
            weights: Layer weights
            
        Returns:
            Layer output tensor
        """
        # Simulate layer computation based on type
        # In practice, this would call actual GPU kernels
        
        if layer_info.layer_type == LayerType.LINEAR:
            # Matrix multiplication for linear layer
            # Simulate compute time proportional to operations
            compute_time = np.prod(layer_info.weight_shape) * 1e-9
            time.sleep(compute_time)
            
            # Dummy computation: matrix multiply
            batch_size = input_data.shape[0] if len(input_data.shape) > 1 else 1
            input_dim = weights.shape[1] if len(weights.shape) > 1 else weights.shape[0]
            output_dim = weights.shape[0] if len(weights.shape) > 1 else 1
            
            output = np.random.randn(batch_size, output_dim).astype(np.float32)
            
        elif layer_info.layer_type == LayerType.CONV2D:
            # Convolution operation
            compute_time = np.prod(layer_info.weight_shape) * 5e-9
            time.sleep(compute_time)
            
            # Dummy convolution output
            output = np.random.randn(*input_data.shape).astype(np.float32)
            
        elif layer_info.layer_type == LayerType.ATTENTION:
            # Attention mechanism (most expensive)
            compute_time = np.prod(layer_info.weight_shape) * 1e-8
            time.sleep(compute_time)
            
            # Dummy attention output
            output = np.random.randn(*input_data.shape).astype(np.float32)
            
        elif layer_info.layer_type == LayerType.EMBEDDING:
            # Embedding lookup
            compute_time = layer_info.weight_size_bytes * 1e-10
            time.sleep(compute_time)
            
            # Dummy embedding output
            embed_dim = weights.shape[1] if len(weights.shape) > 1 else weights.shape[0]
            seq_len = input_data.shape[1] if len(input_data.shape) > 1 else input_data.shape[0]
            output = np.random.randn(1, seq_len, embed_dim).astype(np.float32)
            
        else:
            # Default case - minimal computation
            compute_time = 1e-6  # 1 microsecond
            time.sleep(compute_time)
            output = input_data  # Pass through
        
        return output
    
    def batch_inference(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """
        Execute batch of inference requests
        
        Args:
            requests: List of inference requests
            
        Returns:
            List of inference results
        """
        results = []
        
        # Sort requests by priority
        sorted_requests = sorted(requests, key=lambda x: x.priority)
        
        for request in sorted_requests:
            result = self.inference(request)
            results.append(result)
        
        return results
    
    def benchmark_throughput(self, model_name: str, batch_sizes: List[int],
                           num_iterations: int = 10) -> Dict[str, List[float]]:
        """
        Benchmark inference throughput for different batch sizes
        
        Args:
            model_name: Name of model to benchmark
            batch_sizes: List of batch sizes to test
            num_iterations: Number of iterations per batch size
            
        Returns:
            Throughput measurements
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        model_config = self.models[model_name]
        results = {'batch_sizes': batch_sizes, 'latencies': [], 'throughputs': []}
        
        for batch_size in batch_sizes:
            print(f"Benchmarking batch size {batch_size}...")
            
            latencies = []
            throughputs = []
            
            for i in range(num_iterations):
                # Create dummy input
                input_shape = [batch_size, 512]  # Assume 512-dim input
                input_data = np.random.randn(*input_shape).astype(np.float32)
                
                request = InferenceRequest(
                    request_id=f"bench_{batch_size}_{i}",
                    input_data=input_data,
                    model_name=model_name,
                    timestamp=time.time()
                )
                
                result = self.inference(request)
                latencies.append(result.latency_ms)
                throughputs.append(result.throughput_tokens_per_sec)
            
            avg_latency = np.mean(latencies)
            avg_throughput = np.mean(throughputs)
            
            results['latencies'].append(avg_latency)
            results['throughputs'].append(avg_throughput)
            
            print(f"  Average latency: {avg_latency:.1f}ms")
            print(f"  Average throughput: {avg_throughput:.1f} tokens/sec")
        
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'engine': self.stats.copy(),
            'memory_manager': self.memory_manager.get_stats(),
            'local_cache': self.local_cache.get_stats(),
            'prefetcher': self.prefetcher.get_stats(),
            'cxl_emulator': self.cxl_emulator.get_performance_stats() if self.cxl_emulator else None
        }
    
    def create_sample_transformer_model(self, num_layers: int = 12, 
                                      hidden_size: int = 768,
                                      vocab_size: int = 50000) -> Tuple[ModelConfig, Dict[str, np.ndarray]]:
        """
        Create a sample transformer model for testing
        
        Args:
            num_layers: Number of transformer layers
            hidden_size: Hidden dimension size
            vocab_size: Vocabulary size
            
        Returns:
            Model configuration and weights
        """
        layers = []
        weights = {}
        total_params = 0
        
        # Embedding layer
        embed_shape = (vocab_size, hidden_size)
        embed_size = np.prod(embed_shape) * 4  # 4 bytes per float32
        
        layers.append(LayerInfo(
            name="embedding",
            layer_type=LayerType.EMBEDDING,
            weight_shape=embed_shape,
            weight_size_bytes=embed_size,
            computation_time_ms=1.0,
            reuse_frequency=10
        ))
        
        weights["embedding"] = np.random.randn(*embed_shape).astype(np.float32)
        total_params += np.prod(embed_shape)
        
        # Transformer layers
        for i in range(num_layers):
            # Self-attention weights
            attn_shape = (hidden_size, hidden_size * 3)  # Q, K, V combined
            attn_size = np.prod(attn_shape) * 4
            
            layers.append(LayerInfo(
                name=f"layer_{i}_attention",
                layer_type=LayerType.ATTENTION,
                weight_shape=attn_shape,
                weight_size_bytes=attn_size,
                computation_time_ms=10.0,
                reuse_frequency=1
            ))
            
            weights[f"layer_{i}_attention"] = np.random.randn(*attn_shape).astype(np.float32)
            total_params += np.prod(attn_shape)
            
            # Feed-forward weights
            ff1_shape = (hidden_size, hidden_size * 4)
            ff2_shape = (hidden_size * 4, hidden_size)
            
            for ff_layer, ff_shape in [("ff1", ff1_shape), ("ff2", ff2_shape)]:
                ff_size = np.prod(ff_shape) * 4
                
                layers.append(LayerInfo(
                    name=f"layer_{i}_{ff_layer}",
                    layer_type=LayerType.LINEAR,
                    weight_shape=ff_shape,
                    weight_size_bytes=ff_size,
                    computation_time_ms=5.0,
                    reuse_frequency=1
                ))
                
                weights[f"layer_{i}_{ff_layer}"] = np.random.randn(*ff_shape).astype(np.float32)
                total_params += np.prod(ff_shape)
            
            # Layer normalization
            ln_shape = (hidden_size,)
            ln_size = np.prod(ln_shape) * 4
            
            for ln_layer in ["ln1", "ln2"]:
                layers.append(LayerInfo(
                    name=f"layer_{i}_{ln_layer}",
                    layer_type=LayerType.NORMALIZATION,
                    weight_shape=ln_shape,
                    weight_size_bytes=ln_size,
                    computation_time_ms=0.5,
                    reuse_frequency=1
                ))
                
                weights[f"layer_{i}_{ln_layer}"] = np.random.randn(*ln_shape).astype(np.float32)
                total_params += np.prod(ln_shape)
        
        # Output head
        head_shape = (hidden_size, vocab_size)
        head_size = np.prod(head_shape) * 4
        
        layers.append(LayerInfo(
            name="output_head",
            layer_type=LayerType.LINEAR,
            weight_shape=head_shape,
            weight_size_bytes=head_size,
            computation_time_ms=8.0,
            reuse_frequency=1
        ))
        
        weights["output_head"] = np.random.randn(*head_shape).astype(np.float32)
        total_params += np.prod(head_shape)
        
        total_size_mb = (total_params * 4) / (1024 * 1024)
        
        model_config = ModelConfig(
            name=f"transformer_{num_layers}L_{hidden_size}H",
            layers=layers,
            total_params=total_params,
            total_size_mb=total_size_mb,
            architecture="transformer"
        )
        
        return model_config, weights