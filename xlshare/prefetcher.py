"""
Model-Aware Prefetching Algorithm for XL-Share

Implements intelligent prefetching based on neural network computation graphs
to overlap communication and computation.
"""

import time
import threading
import queue
import simpy
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np


class LayerType(Enum):
    """Types of neural network layers"""
    LINEAR = "linear"
    CONV2D = "conv2d" 
    ATTENTION = "attention"
    EMBEDDING = "embedding"
    NORMALIZATION = "normalization"
    ACTIVATION = "activation"


@dataclass
class LayerInfo:
    """Information about a neural network layer"""
    name: str
    layer_type: LayerType
    weight_shape: Tuple[int, ...]
    weight_size_bytes: int
    computation_time_ms: float
    memory_access_pattern: str = "sequential"
    reuse_frequency: int = 1
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class PrefetchTask:
    """Represents a prefetch task"""
    layer_name: str
    weight_address: int
    weight_size: int
    priority: int
    issue_time: float
    completion_time: Optional[float] = None
    
    def __lt__(self, other):
        return self.priority < other.priority


class ModelAwarePrefetcher:
    """
    Intelligent prefetcher that analyzes neural network computation graphs
    to optimize weight transfers from CXL memory to local GPU cache.
    """
    
    def __init__(self, memory_manager, local_cache, prefetch_threads: int = 2, env=None):
        """
        Initialize model-aware prefetcher
        
        Args:
            memory_manager: CXL memory manager instance
            local_cache: Local GPU cache instance
            prefetch_threads: Number of prefetch worker threads
        """
        self.memory_manager = memory_manager
        self.local_cache = local_cache
        self.env = env
        
        # Model topology and scheduling
        self.layers: Dict[str, LayerInfo] = {}
        self.weight_addresses: Dict[str, int] = {}
        self.execution_order: List[str] = []
        
        # Prefetch queue and workers
        if self.env:
            self.prefetch_queue = simpy.Store(self.env)
            self.active_prefetches: Dict[str, simpy.Event] = {}
            for i in range(prefetch_threads):
                self.env.process(self._prefetch_worker())
        else:
            self.prefetch_queue = queue.PriorityQueue()
            self.active_prefetches: Dict[str, PrefetchTask] = {}
            self.prefetch_workers = []
            self.shutdown_flag = threading.Event()
            for i in range(prefetch_threads):
                worker = threading.Thread(target=self._prefetch_worker_thread, daemon=True)
                worker.start()
                self.prefetch_workers.append(worker)

        # Performance tracking
        self.stats = {
            'prefetch_requests': 0,
            'prefetch_hits': 0,
            'prefetch_misses': 0,
            'cache_stalls': 0,
            'overlap_efficiency': 0.0,
            'bandwidth_utilization': 0.0
        }
        
        print(f"Model-aware prefetcher initialized with {prefetch_threads} workers")
        
        # Adaptive prefetching state
        self.current_lookahead = 2
        self.min_lookahead = 1
        self.max_lookahead = 10
        self.adaptation_interval = 5
        self.steps_since_adaptation = 0
        self.last_stats = self.stats.copy()
    
    def register_model(self, layers: List[LayerInfo], weight_addresses: Dict[str, int]):
        """
        Register neural network model for prefetch optimization
        
        Args:
            layers: List of layer information
            weight_addresses: Mapping of layer names to CXL addresses
        """
        self.layers = {layer.name: layer for layer in layers}
        self.weight_addresses = weight_addresses.copy()
        
        # Determine execution order based on dependencies
        self.execution_order = self._topological_sort(layers)
        
        # Analyze access patterns and compute prefetch priorities
        self._analyze_access_patterns()
        
        print(f"Registered model with {len(layers)} layers")
        print(f"Execution order: {' -> '.join(self.execution_order[:5])}...")
    
    def _topological_sort(self, layers: List[LayerInfo]) -> List[str]:
        """
        Topological sort of layers based on dependencies
        
        Args:
            layers: List of layer information
            
        Returns:
            Execution order of layers
        """
        # Simple implementation - assume sequential order for now
        # In practice, this would do proper topological sorting
        return [layer.name for layer in sorted(layers, key=lambda x: x.name)]
    
    def _analyze_access_patterns(self):
        """Analyze model access patterns for optimization"""
        for layer_name in self.execution_order:
            layer = self.layers[layer_name]
            
            # Estimate computation time based on layer type and size
            if layer.layer_type == LayerType.LINEAR:
                # Linear layer: time proportional to matrix multiply
                layer.computation_time_ms = np.prod(layer.weight_shape) / 1e6
            elif layer.layer_type == LayerType.CONV2D:
                # Convolution: more expensive
                layer.computation_time_ms = np.prod(layer.weight_shape) / 5e5
            elif layer.layer_type == LayerType.ATTENTION:
                # Attention: most expensive
                layer.computation_time_ms = np.prod(layer.weight_shape) / 1e5
            else:
                # Default estimate
                layer.computation_time_ms = layer.weight_size_bytes / 1e6
            
            # Set reuse frequency based on layer type
            if layer.layer_type in [LayerType.EMBEDDING, LayerType.NORMALIZATION]:
                layer.reuse_frequency = 10  # High reuse
            elif layer.layer_type == LayerType.ATTENTION:
                layer.reuse_frequency = 3   # Medium reuse
            else:
                layer.reuse_frequency = 1   # Single use
    
    def schedule_prefetch(self, layer_name: str, priority: int = 0):
        """
        Schedule prefetch for a layer's weights
        
        Args:
            layer_name: Name of layer to prefetch
            priority: Prefetch priority (lower = higher priority)
        """
        if layer_name not in self.weight_addresses:
            return False
        
        # Check if already prefetched or in progress
        if (self.local_cache.get(layer_name) is not None or 
            layer_name in self.active_prefetches):
            return True
        
        layer = self.layers[layer_name]
        weight_address = self.weight_addresses[layer_name]
        
        task = PrefetchTask(
            layer_name=layer_name,
            weight_address=weight_address,
            weight_size=layer.weight_size_bytes,
            priority=priority,
            issue_time=self.env.now if self.env else time.time()
        )
        
        if self.env:
            self.active_prefetches[layer_name] = self.env.event()
            self.prefetch_queue.put(task)
        else:
            self.prefetch_queue.put(task)
            self.active_prefetches[layer_name] = task
        self.stats['prefetch_requests'] += 1
        
        return True
    
    def smart_prefetch_pipeline(self, current_layer_idx: int, lookahead: Optional[int] = None):
        """
        Intelligently prefetch upcoming layers based on computation pipeline.
        Adapts lookahead dynamically if no explicit lookahead provided.
        
        Args:
            current_layer_idx: Index of currently executing layer
            lookahead: Number of layers to prefetch ahead (optional)
        """
        # Dynamic Adaptation
        if lookahead is None:
            self.steps_since_adaptation += 1
            if self.steps_since_adaptation >= self.adaptation_interval:
                self._adapt_strategy()
                self.steps_since_adaptation = 0
            lookahead = self.current_lookahead

        # Prefetch next layers with decreasing priority
        for i in range(1, min(lookahead + 1, len(self.execution_order) - current_layer_idx)):
            next_layer_idx = current_layer_idx + i
            next_layer_name = self.execution_order[next_layer_idx]
            
            # Calculate priority based on distance and layer importance
            priority = i * 10  # Lower priority for farther layers
            
            # Adjust priority based on layer characteristics
            next_layer = self.layers[next_layer_name]
            if next_layer.layer_type == LayerType.ATTENTION:
                priority -= 5  # Higher priority for expensive layers
            if next_layer.reuse_frequency > 1:
                priority -= 3  # Higher priority for reused weights
            
            self.schedule_prefetch(next_layer_name, priority)
            
        # [NEW] Graph-Aware Eviction Update
        # Update the cache with distances to next usage for ALL layers
        # This is O(N) per step, but N is small (layers ~100)
        future_map = {}
        for idx in range(current_layer_idx, len(self.execution_order)):
             layer_name = self.execution_order[idx]
             if layer_name not in future_map:
                 future_map[layer_name] = idx
        
        # Items not in this list (past) get implicit infinity if not in future_map
        # But we only update what we know.
        self.local_cache.update_future_accesses(future_map)
    
    def wait_for_weights(self, layer_name: str, timeout: float = 5.0):
        """
        Wait for layer weights to be available in cache
        
        Args:
            layer_name: Name of layer
            timeout: Maximum wait time in seconds
            
        Returns:
            Layer weights as numpy array
        """
        # First check if already in cache
        weights = self.local_cache.get(layer_name)
        if weights is not None:
            self.stats['prefetch_hits'] += 1
            return self._deserialize_weights(weights, self.layers[layer_name].weight_shape)

        if self.env:
            if layer_name in self.active_prefetches:
                yield self.active_prefetches[layer_name]
                weights = self.local_cache.get(layer_name)
                if weights is not None:
                    self.stats['prefetch_hits'] += 1
                    return self._deserialize_weights(weights, self.layers[layer_name].weight_shape)
        else:
            start_time = time.time()
            # Wait for prefetch to complete
            while time.time() - start_time < timeout:
                if layer_name in self.active_prefetches:
                    task = self.active_prefetches[layer_name]
                    if task.completion_time is not None:
                        weights = self.local_cache.get(layer_name)
                        if weights is not None:
                            self.stats['prefetch_hits'] += 1
                            return self._deserialize_weights(weights, self.layers[layer_name].weight_shape)
                
                time.sleep(0.001)  # 1ms polling
        
        # Prefetch failed or timed out - fetch directly from CXL memory
        self.stats['prefetch_misses'] += 1
        self.stats['cache_stalls'] += 1
        
        weights = yield from self._fetch_weights_direct(layer_name)
        return weights
    
    def _fetch_weights_direct(self, layer_name: str):
        """
        Directly fetch weights from CXL memory (cache miss)
        
        Args:
            layer_name: Name of layer
            
        Returns:
            Layer weights
        """
        layer = self.layers[layer_name]
        address = self.weight_addresses[layer_name]
        
        # Read from CXL memory
        if self.env:
            weight_bytes = yield self.env.process(self.memory_manager.read(address, layer.weight_size_bytes))
        else:
            weight_bytes = self.memory_manager.read(address, layer.weight_size_bytes)
        
        # Store in cache for future use
        self.local_cache.put(layer_name, weight_bytes)
        
        return self._deserialize_weights(weight_bytes, layer.weight_shape)
    
    def _deserialize_weights(self, weight_bytes: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Convert serialized bytes back to weight tensor
        
        Args:
            weight_bytes: Serialized weight data
            shape: Original weight shape
            
        Returns:
            Deserialized weight tensor
        """
        flat_weights = np.frombuffer(weight_bytes.tobytes(), dtype=np.float32)
        return flat_weights.reshape(shape)
    
    def _prefetch_worker(self):
        """Worker process for handling prefetch requests"""
        while True:
            task = yield self.prefetch_queue.get()
            
            # Fetch weights from CXL memory
            weight_bytes = yield self.env.process(
                self.memory_manager.read(task.weight_address, task.weight_size)
            )
            
            # Store in local cache
            layer = self.layers[task.layer_name]
            pin_in_cache = layer.reuse_frequency > 1
            
            self.local_cache.put(
                task.layer_name, 
                weight_bytes, 
                pin=pin_in_cache
            )
            
            # Mark task as completed
            task.completion_time = self.env.now
            self.active_prefetches[task.layer_name].succeed()
            
            # Update statistics
            transfer_time = task.completion_time - task.issue_time
            bandwidth_gbps = (task.weight_size / (1024**3)) / (transfer_time / 1e9)
            self.stats['bandwidth_utilization'] = bandwidth_gbps

    def _prefetch_worker_thread(self):
        """Worker thread for handling prefetch requests"""
        while not self.shutdown_flag.is_set():
            try:
                # Get next prefetch task
                task = self.prefetch_queue.get(timeout=1.0)
                
                # Fetch weights from CXL memory
                weight_bytes = self.memory_manager.read(
                    task.weight_address, 
                    task.weight_size
                )
                
                # Store in local cache
                layer = self.layers[task.layer_name]
                pin_in_cache = layer.reuse_frequency > 1
                
                self.local_cache.put(
                    task.layer_name, 
                    weight_bytes, 
                    pin=pin_in_cache
                )
                
                # Mark task as completed
                task.completion_time = time.time()
                
                # Update statistics
                transfer_time = task.completion_time - task.issue_time
                bandwidth_gbps = (task.weight_size / (1024**3)) / transfer_time
                self.stats['bandwidth_utilization'] = bandwidth_gbps
                
                self.prefetch_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Prefetch worker error: {e}")
                continue
    
    def get_prefetch_efficiency(self) -> float:
        """
        Calculate prefetch efficiency metric
        
        Returns:
            Efficiency ratio (0.0 to 1.0)
        """
        total_requests = self.stats['prefetch_hits'] + self.stats['prefetch_misses']
        if total_requests == 0:
            return 0.0
        
        return self.stats['prefetch_hits'] / total_requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prefetcher statistics"""
        stats = self.stats.copy()
        stats['efficiency'] = self.get_prefetch_efficiency()
        stats['active_prefetches'] = len(self.active_prefetches)
        if self.env:
            stats['queue_depth'] = len(self.prefetch_queue.items)
        else:
            stats['queue_depth'] = self.prefetch_queue.qsize()
        
        return stats
    
    def _adapt_strategy(self):
        """Adapt lookahead based on recent performance metrics"""
        current_stalls = self.stats['cache_stalls'] - self.last_stats['cache_stalls']
        current_bw = self.stats['bandwidth_utilization'] # Instantaneous
        
        # Heuristic:
        # If we are stalling, we need to prefetch earlier (increase lookahead),
        # UNLESS we are already saturating bandwidth.
        
        if current_stalls > 0:
            if current_bw < 60.0: # Assuming 64GB/s max in default profile
                 self.current_lookahead = min(self.current_lookahead + 1, self.max_lookahead)
                 print(f"[Adaptive] Stalls detected ({current_stalls}), increasing lookahead to {self.current_lookahead}")
            else:
                 print(f"[Adaptive] Stalls detected but bandwidth saturated ({current_bw:.1f} GB/s). Keeping lookahead {self.current_lookahead}")
        
        # If we have NO stalls and high bandwidth usage, maybe we are too aggressive?
        # Or if we have no stalls and low bandwidth, we are fine.
        elif current_bw > 60.0:
             # Reduce pressure
             self.current_lookahead = max(self.current_lookahead - 1, self.min_lookahead)
             print(f"[Adaptive] Bandwidth saturated, reducing lookahead to {self.current_lookahead}")

        self.last_stats = self.stats.copy()

    def shutdown(self):
        """Shutdown prefetcher and worker threads"""
        self.shutdown_flag.set()
        for worker in self.prefetch_workers:
            worker.join(timeout=1.0)