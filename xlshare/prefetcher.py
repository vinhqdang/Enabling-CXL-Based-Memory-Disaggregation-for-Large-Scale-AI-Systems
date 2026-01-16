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
        self.current_lookahead = 5  # [TUNING] Start with aggressive lookahead
        self.min_lookahead = 2
        self.max_lookahead = 20
        self.adaptation_interval = 2 # [TUNING] Adapt faster
        self.steps_since_adaptation = 0
        self.last_stats = self.stats.copy()

        # [NEW] Advanced Baselines Configuration
        self.mode = "camp" # Options: no_prefetch, static, tmo, melody, limoncello, expand, camp
        self.debug = False
        self.history_window_ms = 300.0 # Window for history replay
    
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
        
        # [NEW] Static Pinning Analysis
        # Identify layers that fit in cache to pin them permanently (Tiered Caching)
        # Prioritize High Reuse Frequency layers
        current_usage = 0
        self.pinned_layers = set()
        print(f"Cache Capacity: {self.local_cache.capacity} bytes")
        
        # Sort layers by importance (Frequency DESC, then original order)
        candidates = sorted(self.execution_order, key=lambda x: self.layers[x].reuse_frequency, reverse=True)
        
        for layer_name in candidates:
            layer = self.layers[layer_name]
            layer_size = layer.weight_size_bytes
            # [TUNING] Increase reservation to 90% for constrained scenarios to maximizing pinning
            if current_usage + layer_size < self.local_cache.capacity * 0.90: 
                self.pinned_layers.add(layer_name)
                current_usage += layer_size
            else:
                # If high frequency but doesn't fit, we stop? 
                # Or continue checking smaller layers? Continue.
                continue
                
        print(f"Registered model with {len(layers)} layers")
        print(f"Pinned Layers ({len(self.pinned_layers)}): {str(list(self.pinned_layers)[:3])}...")
        print(f"Execution order: {' -> '.join(self.execution_order[:5])}...")
    
    def _topological_sort(self, layers: List[LayerInfo]) -> List[str]:
        """
        Topological sort of layers based on dependencies
        """
        # Simple implementation - assume sequential order for now
        return [layer.name for layer in sorted(layers, key=lambda x: x.name)]
    
    def _analyze_access_patterns(self):
        """Analyze model access patterns for optimization"""
        for layer_name in self.execution_order:
            layer = self.layers[layer_name]
            
            # Estimate computation time based on layer type and size
            # [FIX] Preserve manual overrides
            if layer.computation_time_ms > 0 and not layer.name.startswith("layer_"):
                 pass # Keep existing value
            elif layer.layer_type == LayerType.LINEAR:
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
            
            # Set reuse frequency based on layer type (if not manually set)
            if layer.reuse_frequency > 1:
                pass # Respect manual setting
            elif layer.layer_type in [LayerType.EMBEDDING, LayerType.NORMALIZATION]:
                layer.reuse_frequency = 10  # High reuse
            elif layer.layer_type == LayerType.ATTENTION:
                layer.reuse_frequency = 3   # Medium reuse
            else:
                layer.reuse_frequency = 1   # Single use
    
    def schedule_prefetch(self, layer_name: str, priority: int = 0):
        """
        Schedule prefetch for a layer's weights
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
        """
        # Dynamic Adaptation
        if lookahead is None:
            if self.mode == "expand":
                # [BASELINE 3] ExPAND (IEEE Micro '25) - Expander/History Based
                # Uses "Heterogeneous Address Prediction" (Oracle trace replay here).
                accumulated_time = 0.0
                k = 0
                for i in range(1, len(self.execution_order) - current_layer_idx):
                     next_idx = current_layer_idx + i
                     next_name = self.execution_order[next_idx]
                     accumulated_time += self.layers[next_name].computation_time_ms
                     k += 1
                     if accumulated_time > self.history_window_ms:
                         break
                lookahead = max(k, self.min_lookahead)
                lookahead = min(lookahead, self.max_lookahead)
                
            elif self.mode == "limoncello":
                # [BASELINE 4] Limoncello (ASPLOS '24) - Targeted Software Prefetching
                lookahead = 10 
                
            else:
                # TMO / Melody / Static / CAMP use the state variable
                self.steps_since_adaptation += 1
                if self.steps_since_adaptation >= self.adaptation_interval:
                    self._adapt_strategy()
                    self.steps_since_adaptation = 0
                
                # [OPTIMIZATION] CAMP Semantic Opportunity Seizing
                # If we are in a "Thinking Phase" (Long Compute), we should NOT be conservative.
                # We should seize the opportunity to fill the buffer immediately.
                if self.mode == "camp":
                    current_layer_name = self.execution_order[current_layer_idx]
                    
                    if self.layers[current_layer_name].computation_time_ms > 50.0:
                         # [SAFETY] Cache-Aware Flow Control
                         # Don't prefetch more than cache can hold.
                         # Calculate how many future layers fit in remaining capacity.
                         current_usage = self.local_cache.current_size
                         pending_bytes = 0
                         safe_k = 0
                         
                         # Check active (pending) prefetches first
                         # (Approximation: assume they are committed)
                         
                         for i in range(1, self.max_lookahead + 1):
                             if current_layer_idx + i >= len(self.execution_order): break
                             next_name = self.execution_order[current_layer_idx + i]
                             layer_size = self.layers[next_name].weight_size_bytes
                             
                             if current_usage + pending_bytes + layer_size < self.local_cache.capacity:
                                 safe_k = i
                                 pending_bytes += layer_size
                             else:
                                 break
                         
                         if self.debug: print(f"[CAMP] Semantic Opportunity! Safe to prefetch {safe_k} layers ({pending_bytes/1e6:.1f} MB)")
                         lookahead = max(self.current_lookahead, safe_k)
                    else:
                         lookahead = self.current_lookahead
                else:
                    lookahead = self.current_lookahead


        # Clear stale future access info
        self.local_cache.future_accesses.clear()
        
        # [CRITICAL] Re-assert pinned layers immediately
        for pinned in self.pinned_layers:
            self.local_cache.future_accesses[pinned] = -999

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
            
            # Inform Cache about future access distance (for Graph-Aware Eviction)
            self.local_cache.future_accesses[next_layer_name] = i
        
        # Look further ahead for eviction planning
        eviction_horizon = max(lookahead * 2, 20)
        
        for i in range(1, eviction_horizon + 1):
             next_layer_idx = (current_layer_idx + i) % len(self.execution_order)
             next_layer_name = self.execution_order[next_layer_idx]
             
             if next_layer_name in self.pinned_layers:
                 # PINNED: Distance is effectively negative (infinite value)
                 self.local_cache.future_accesses[next_layer_name] = -999
             elif next_layer_name not in self.local_cache.future_accesses:
                 self.local_cache.future_accesses[next_layer_name] = i
            
        # [NEW] Graph-Aware Eviction Update
        # [OPTIMIZATION] Limit scan to horizon to reduce Python overhead for fast/streaming scenarios
        future_map = {}
        scan_limit = min(current_layer_idx + eviction_horizon + 10, len(self.execution_order))
        for idx in range(current_layer_idx, scan_limit):
             layer_name = self.execution_order[idx]
             if layer_name not in future_map:
                 future_map[layer_name] = idx
        
        self.local_cache.update_future_accesses(future_map)
    
    def wait_for_weights(self, layer_name: str, timeout: float = 5.0):
        """
        Wait for layer weights to be available in cache
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
        
        if self.mode == "tmo":
             # [BASELINE 1] TMO (ASPLOS '22) - Transparent Memory Offloading
             if current_stalls > 0:
                 self.current_lookahead = max(self.min_lookahead, self.current_lookahead // 2)
                 if self.debug: print(f"[TMO] Pressure detected ({current_stalls}), BACKOFF to {self.current_lookahead}")
             else:
                 self.current_lookahead = min(self.current_lookahead + 1, self.max_lookahead)
                 
        elif self.mode == "melody":
             # [BASELINE 2] Melody (ASPLOS '25) - Systematic Characterization
             if current_bw > 55.0: # ~85% of 64GB/s
                 self.current_lookahead = max(self.min_lookahead, self.current_lookahead - 1)
                 if self.debug: print(f"[Melody] Congestion imminent ({current_bw:.1f} GB/s), THROTTLING to {self.current_lookahead}")
             elif current_stalls > 0:
                 self.current_lookahead = min(self.current_lookahead + 1, self.max_lookahead)
             else:
                 self.current_lookahead = min(self.current_lookahead + 1, self.max_lookahead)

        elif self.mode == "camp":
            # [OUR METHOD] CAMP (Content-Aware Memory Prefetching)
            if current_stalls > 0:
                if current_bw < 60.0: 
                     # [TUNING] Exponential increase on stalls
                     self.current_lookahead = min(self.current_lookahead * 2, self.max_lookahead)
                     if self.debug: print(f"[CAMP] Stalls detected ({current_stalls}), BOOSTING lookahead to {self.current_lookahead}")
                else:
                     if self.debug: print(f"[CAMP] Stalls detected but bandwidth saturated ({current_bw:.1f} GB/s). Keeping lookahead {self.current_lookahead}")
            elif current_bw > 60.0:
                 # Reduce pressure cautiously
                 self.current_lookahead = max(self.current_lookahead - 1, self.min_lookahead)
                 if self.debug: print(f"[CAMP] Bandwidth saturated, trimming lookahead to {self.current_lookahead}")

        self.last_stats = self.stats.copy()

    def shutdown(self):
        """Shutdown prefetcher and worker threads"""
        self.shutdown_flag.set()
        for worker in self.prefetch_workers:
            worker.join(timeout=1.0)