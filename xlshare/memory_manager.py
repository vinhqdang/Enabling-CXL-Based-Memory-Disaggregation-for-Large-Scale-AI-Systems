"""
Memory Management Components for XL-Share System

Implements CXL memory management with local caching and coherence support.
"""

import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from dataclasses import dataclass
import hashlib


@dataclass
class MemoryRegion:
    """Represents a memory region in the CXL pool"""
    address: int
    size: int
    data: np.ndarray
    reference_count: int = 0
    last_accessed: float = 0.0
    is_dirty: bool = False
    
    def __post_init__(self):
        self.last_accessed = time.time()


class CXLMemoryManager:
    """
    Manages CXL-attached memory pool for model parameters.
    
    Provides hardware-coherent distributed shared memory abstraction
    for AI model weights and parameters.
    """
    
    def __init__(self, pool_size_gb: float = 64.0, latency_ns: int = 300):
        """
        Initialize CXL memory manager
        
        Args:
            pool_size_gb: Size of memory pool in GB
            latency_ns: CXL access latency in nanoseconds  
        """
        self.pool_size = int(pool_size_gb * 1024**3)  # Convert to bytes
        self.latency_ns = latency_ns
        
        # Memory pool storage (simulated as dict)
        self.memory_pool: Dict[int, MemoryRegion] = {}
        self.next_address = 0x10000000  # Start at 256MB offset
        
        # Coherence and synchronization
        self.coherence_lock = threading.RLock()
        self.reference_counts: Dict[int, int] = {}
        
        # Statistics
        self.stats = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'total_reads': 0,
            'total_writes': 0,
            'bytes_allocated': 0,
            'bytes_read': 0,
            'bytes_written': 0
        }
        
        print(f"CXL Memory Manager initialized: {pool_size_gb}GB pool, {latency_ns}ns latency")
    
    def allocate(self, size: int, alignment: int = 64) -> int:
        """
        Allocate memory region in CXL pool
        
        Args:
            size: Size in bytes to allocate
            alignment: Memory alignment requirement
            
        Returns:
            Memory address of allocated region
        """
        with self.coherence_lock:
            # Align address
            aligned_addr = (self.next_address + alignment - 1) // alignment * alignment
            
            if aligned_addr + size > self.pool_size:
                raise MemoryError(f"Cannot allocate {size} bytes: pool exhausted")
            
            # Create memory region
            region = MemoryRegion(
                address=aligned_addr,
                size=size,
                data=np.zeros(size, dtype=np.uint8)
            )
            
            self.memory_pool[aligned_addr] = region
            self.reference_counts[aligned_addr] = 1
            self.next_address = aligned_addr + size
            
            # Update statistics
            self.stats['total_allocations'] += 1
            self.stats['bytes_allocated'] += size
            
            return aligned_addr
    
    def deallocate(self, address: int) -> bool:
        """
        Deallocate memory region
        
        Args:
            address: Address to deallocate
            
        Returns:
            True if successfully deallocated
        """
        with self.coherence_lock:
            if address not in self.memory_pool:
                return False
            
            region = self.memory_pool[address]
            self.reference_counts[address] -= 1
            
            if self.reference_counts[address] <= 0:
                del self.memory_pool[address]
                del self.reference_counts[address]
                self.stats['total_deallocations'] += 1
                
            return True
    
    def read(self, address: int, size: int) -> np.ndarray:
        """
        Read data from CXL memory with simulated latency
        
        Args:
            address: Memory address to read from
            size: Number of bytes to read
            
        Returns:
            Data as numpy array
        """
        # Simulate CXL access latency
        time.sleep(self.latency_ns / 1e9)
        
        with self.coherence_lock:
            if address not in self.memory_pool:
                raise ValueError(f"Invalid memory address: 0x{address:x}")
            
            region = self.memory_pool[address]
            region.last_accessed = time.time()
            
            # Update statistics
            self.stats['total_reads'] += 1
            self.stats['bytes_read'] += size
            
            return region.data[:size].copy()
    
    def write(self, address: int, data: np.ndarray) -> bool:
        """
        Write data to CXL memory with coherence
        
        Args:
            address: Memory address to write to
            data: Data to write
            
        Returns:
            True if write successful
        """
        # Simulate CXL access latency
        time.sleep(self.latency_ns / 1e9)
        
        with self.coherence_lock:
            if address not in self.memory_pool:
                raise ValueError(f"Invalid memory address: 0x{address:x}")
            
            region = self.memory_pool[address]
            
            if len(data) > region.size:
                raise ValueError("Data size exceeds allocated region")
            
            region.data[:len(data)] = data
            region.is_dirty = True
            region.last_accessed = time.time()
            
            # Update statistics
            self.stats['total_writes'] += 1
            self.stats['bytes_written'] += len(data)
            
            return True
    
    def store_model_weights(self, weights: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Store model weights in CXL memory pool
        
        Args:
            weights: Dictionary of layer name to weight arrays
            
        Returns:
            Dictionary mapping layer names to CXL addresses
        """
        weight_addresses = {}
        
        for layer_name, weight_array in weights.items():
            # Flatten and convert to bytes
            flat_weights = weight_array.flatten().astype(np.float32)
            weight_bytes = flat_weights.tobytes()
            
            # Allocate memory and store
            address = self.allocate(len(weight_bytes))
            weight_data = np.frombuffer(weight_bytes, dtype=np.uint8)
            self.write(address, weight_data)
            
            weight_addresses[layer_name] = address
            
            print(f"Stored {layer_name}: {weight_array.shape} -> 0x{address:x} ({len(weight_bytes)} bytes)")
        
        return weight_addresses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics"""
        with self.coherence_lock:
            stats = self.stats.copy()
            stats['active_regions'] = len(self.memory_pool)
            stats['total_pool_usage'] = sum(r.size for r in self.memory_pool.values())
            stats['pool_utilization'] = stats['total_pool_usage'] / self.pool_size
            
        return stats


class LocalCache:
    """
    Local GPU memory cache for frequently accessed weights.
    
    Implements LRU eviction policy with model-aware hints.
    """
    
    def __init__(self, capacity_mb: int = 512):
        """
        Initialize local cache
        
        Args:
            capacity_mb: Cache capacity in megabytes
        """
        self.capacity = capacity_mb * 1024 * 1024  # Convert to bytes
        self.cache: OrderedDict[str, Tuple[np.ndarray, int]] = OrderedDict()
        self.current_size = 0
        self.cache_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'bytes_cached': 0
        }
        
        print(f"Local cache initialized: {capacity_mb}MB capacity")
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get data from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached data if found, None otherwise
        """
        with self.cache_lock:
            if key in self.cache:
                # Move to end (most recently used)
                data, size = self.cache.pop(key)
                self.cache[key] = (data, size)
                
                self.stats['hits'] += 1
                return data.copy()
            
            self.stats['misses'] += 1
            return None
    
    def put(self, key: str, data: np.ndarray, pin: bool = False):
        """
        Put data in cache with optional pinning
        
        Args:
            key: Cache key
            data: Data to cache
            pin: Whether to pin in cache (prevent eviction)
        """
        with self.cache_lock:
            data_size = data.nbytes
            
            # Remove existing entry if present
            if key in self.cache:
                _, old_size = self.cache.pop(key)
                self.current_size -= old_size
            
            # Evict until we have space
            while (self.current_size + data_size > self.capacity and 
                   len(self.cache) > 0):
                self._evict_lru()
            
            if self.current_size + data_size <= self.capacity:
                self.cache[key] = (data.copy(), data_size)
                self.current_size += data_size
                self.stats['bytes_cached'] += data_size
                
                if pin:
                    # Move pinned items to end to avoid eviction
                    pinned_data = self.cache.pop(key)
                    self.cache[key] = pinned_data
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if self.cache:
            key, (data, size) = self.cache.popitem(last=False)
            self.current_size -= size
            self.stats['evictions'] += 1
    
    def mark_for_eviction(self, key: str):
        """Mark item as candidate for eviction"""
        with self.cache_lock:
            if key in self.cache:
                # Move to front (least recently used)
                data_tuple = self.cache.pop(key)
                # Insert at beginning
                temp_cache = OrderedDict([(key, data_tuple)])
                temp_cache.update(self.cache)
                self.cache = temp_cache
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_accesses = self.stats['hits'] + self.stats['misses']
        if total_accesses == 0:
            return 0.0
        return self.stats['hits'] / total_accesses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.cache_lock:
            stats = self.stats.copy()
            stats['current_size'] = self.current_size
            stats['utilization'] = self.current_size / self.capacity
            stats['hit_rate'] = self.get_hit_rate()
            stats['items_cached'] = len(self.cache)
            
        return stats