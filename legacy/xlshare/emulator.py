"""
CXL Hardware Emulation Layer

Simulates CXL 3.0 memory access characteristics including latency,
bandwidth, and coherence behavior for testing XL-Share system.
"""

import time
import random
import numpy as np
import threading
import simpy
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class CXLProtocol(Enum):
    """CXL protocol types"""
    CXL_IO = "cxl.io"      # PCIe-based I/O
    CXL_CACHE = "cxl.cache" # Cache coherent access
    CXL_MEM = "cxl.mem"     # Memory semantic access


class MESIState(Enum):
    """MESI coherence states"""
    MODIFIED = "modified"
    EXCLUSIVE = "exclusive"
    SHARED = "shared"
    INVALID = "invalid"


@dataclass
class CXLLatencyProfile:
    """CXL memory access latency characteristics"""
    local_dram_ns: int = 80      # Local DRAM latency
    cxl_near_ns: int = 150       # Near CXL memory (same chassis)
    cxl_far_ns: int = 300        # Far CXL memory (different chassis)
    cxl_network_ns: int = 1000   # Networked CXL memory
    
    # Bandwidth characteristics (GB/s)
    local_bandwidth: float = 400.0    # Local HBM bandwidth
    cxl_bandwidth: float = 64.0       # CXL 3.0 bandwidth
    
    # Coherence overhead
    coherence_overhead_ns: int = 50   # Cache coherence protocol overhead


@dataclass  
class CXLTopology:
    """CXL system topology configuration"""
    num_hosts: int = 4
    memory_pools_per_host: int = 1
    pool_size_gb: float = 64.0
    interconnect_type: str = "switched"  # "switched" or "point-to-point"


class CXLEmulator:
    """
    Emulates CXL 3.0 hardware behavior for testing and development.
    
    Simulates memory pooling, coherence protocols, and performance
    characteristics without requiring actual CXL hardware.
    """
    
    def __init__(self, 
                 latency_profile: Optional[CXLLatencyProfile] = None,
                 topology: Optional[CXLTopology] = None,
                 coherence_enabled: bool = True):
        """
        Initialize CXL emulator
        
        Args:
            latency_profile: CXL latency and bandwidth characteristics
            topology: System topology configuration
            coherence_enabled: Enable coherence protocol simulation
        """
        self.env = simpy.Environment()
        self.latency_profile = latency_profile or CXLLatencyProfile()
        self.topology = topology or CXLTopology()
        self.coherence_enabled = coherence_enabled
        
        # Memory pools (simulated)
        self.memory_pools: Dict[int, Dict[int, np.ndarray]] = {}
        self.pool_locks: Dict[int, threading.RLock] = {}
        
        # Coherence state tracking
        self.coherence_state: Dict[Tuple[int, int], MESIState] = {}  # (pool_id, addr) -> state
        self.coherence_lock = threading.RLock()
        
        # Performance tracking
        self.stats = {
            'local_accesses': 0,
            'remote_accesses': 0,
            'coherence_misses': 0,
            'bandwidth_samples': [],
            'latency_samples': []
        }
        self.concurrent_accesses: Dict[int, int] = {}
        
        # Initialize memory pools
        for pool_id in range(self.topology.num_hosts * self.topology.memory_pools_per_host):
            self.memory_pools[pool_id] = {}
            self.pool_locks[pool_id] = threading.RLock()
            self.concurrent_accesses[pool_id] = 0
        
        print(f"CXL Emulator initialized:")
        print(f"  - {self.topology.num_hosts} hosts")
        print(f"  - {len(self.memory_pools)} memory pools")
        print(f"  - Coherence: {'enabled' if coherence_enabled else 'disabled'}")
        print(f"  - CXL latency: {self.latency_profile.cxl_near_ns}ns (near)")

    @classmethod
    def from_profile_dict(cls, profile: dict, **kwargs) -> "CXLEmulator":
        lp = CXLLatencyProfile(
            local_dram_ns=profile.get("local_dram_ns", 80),
            cxl_near_ns=profile.get("cxl_near_ns", 150),
            cxl_far_ns=profile.get("cxl_far_ns", 300),
            cxl_network_ns=profile.get("cxl_network_ns", 1000),
            local_bandwidth=profile.get("local_bandwidth", 400.0),
            cxl_bandwidth=profile.get("cxl_bandwidth", 64.0),
            coherence_overhead_ns=profile.get("coherence_overhead_ns", 50),
        )
        return cls(latency_profile=lp, **kwargs)
    
    def allocate_memory(self, pool_id: int, size: int, alignment: int = 64) -> int:
        """
        Allocate memory in specified CXL pool
        
        Args:
            pool_id: Target memory pool ID
            size: Size in bytes to allocate
            alignment: Memory alignment requirement
            
        Returns:
            Virtual address of allocated memory
        """
        if pool_id not in self.memory_pools:
            raise ValueError(f"Invalid pool ID: {pool_id}")
        
        with self.pool_locks[pool_id]:
            # Find next available address
            used_addresses = set(self.memory_pools[pool_id].keys())
            addr = alignment
            
            while addr in used_addresses:
                addr += alignment
            
            # Allocate memory
            self.memory_pools[pool_id][addr] = np.zeros(size, dtype=np.uint8)
            
            # Initialize coherence state
            if self.coherence_enabled:
                with self.coherence_lock:
                    self.coherence_state[(pool_id, addr)] = MESIState.INVALID
            
            return addr
    
    def read_memory(self, pool_id: int, addr: int, size: int, 
                   source_host: int = 0):
        """
        Read memory with CXL latency simulation
        
        Args:
            pool_id: Memory pool ID
            addr: Memory address
            size: Number of bytes to read
            source_host: Host performing the read
            
        Returns:
            Data as numpy array
        """
        start_time = self.env.now
        
        # Determine access type and latency
        is_local = (pool_id // self.topology.memory_pools_per_host) == source_host
        
        if is_local:
            latency_ns = self.latency_profile.local_dram_ns
            bandwidth_gbps = self.latency_profile.local_bandwidth
            self.stats['local_accesses'] += 1
        else:
            latency_ns = self._calculate_remote_latency(pool_id, source_host)
            bandwidth_gbps = self.latency_profile.cxl_bandwidth
            self.stats['remote_accesses'] += 1
        
        # Add coherence overhead if enabled
        if self.coherence_enabled:
            coherence_miss = self._handle_coherence_read(pool_id, addr, source_host)
            if coherence_miss:
                latency_ns += self.latency_profile.coherence_overhead_ns
                self.stats['coherence_misses'] += 1
        
        # Calculate transfer time based on bandwidth
        transfer_time_s = size / (bandwidth_gbps * 1024**3)
        latency_ns += transfer_time_s * 1e9

        # Simulate network contention
        yield self.env.process(self.simulate_network_contention(pool_id))

        # Simulate access latency
        with self.pool_locks[pool_id]:
            self.concurrent_accesses[pool_id] += 1
        
        yield self.env.timeout(latency_ns)
        
        with self.pool_locks[pool_id]:
            self.concurrent_accesses[pool_id] -= 1

        # Read data from pool
        with self.pool_locks[pool_id]:
            if addr not in self.memory_pools[pool_id]:
                raise ValueError(f"Invalid address: {addr} in pool {pool_id}")
            
            data = self.memory_pools[pool_id][addr][:size].copy()
        
        # Record performance metrics
        transfer_time = self.env.now - start_time
        actual_bandwidth_gbps = (size / (1024**3)) / max(transfer_time / 1e9, 1e-9)
        
        self.stats['bandwidth_samples'].append(actual_bandwidth_gbps)
        self.stats['latency_samples'].append(latency_ns)
        
        return data
    
    def write_memory(self, pool_id: int, addr: int, data: np.ndarray,
                    source_host: int = 0):
        """
        Write memory with CXL latency simulation
        
        Args:
            pool_id: Memory pool ID
            addr: Memory address
            data: Data to write
            source_host: Host performing the write
            
        Returns:
            True if write successful
        """
        start_time = self.env.now
        
        # Determine access type and latency
        is_local = (pool_id // self.topology.memory_pools_per_host) == source_host
        
        if is_local:
            latency_ns = self.latency_profile.local_dram_ns
            bandwidth_gbps = self.latency_profile.local_bandwidth
            self.stats['local_accesses'] += 1
        else:
            latency_ns = self._calculate_remote_latency(pool_id, source_host)
            bandwidth_gbps = self.latency_profile.cxl_bandwidth
            self.stats['remote_accesses'] += 1
        
        # Handle coherence for writes
        if self.coherence_enabled:
            self._handle_coherence_write(pool_id, addr, source_host)
            latency_ns += self.latency_profile.coherence_overhead_ns
        
        # Calculate transfer time based on bandwidth
        transfer_time_s = len(data) / (bandwidth_gbps * 1024**3)
        latency_ns += transfer_time_s * 1e9

        # Simulate network contention
        yield self.env.process(self.simulate_network_contention(pool_id))

        # Simulate access latency
        with self.pool_locks[pool_id]:
            self.concurrent_accesses[pool_id] += 1
        
        yield self.env.timeout(latency_ns)

        with self.pool_locks[pool_id]:
            self.concurrent_accesses[pool_id] -= 1
        
        # Write data to pool
        with self.pool_locks[pool_id]:
            if addr not in self.memory_pools[pool_id]:
                raise ValueError(f"Invalid address: {addr} in pool {pool_id}")
            
            pool_data = self.memory_pools[pool_id][addr]
            if len(data) > len(pool_data):
                raise ValueError("Data size exceeds allocated region")
            
            pool_data[:len(data)] = data
        
        # Record performance metrics
        transfer_time = self.env.now - start_time
        actual_bandwidth_gbps = (len(data) / (1024**3)) / max(transfer_time / 1e9, 1e-9)
        
        self.stats['bandwidth_samples'].append(actual_bandwidth_gbps)
        self.stats['latency_samples'].append(latency_ns)
        
        return True
    
    def _calculate_remote_latency(self, pool_id: int, source_host: int) -> int:
        """
        Calculate latency for remote memory access
        
        Args:
            pool_id: Target memory pool
            source_host: Source host ID
            
        Returns:
            Latency in nanoseconds
        """
        target_host = pool_id // self.topology.memory_pools_per_host
        
        if target_host == source_host:
            return self.latency_profile.local_dram_ns
        elif abs(target_host - source_host) == 1:
            return self.latency_profile.cxl_near_ns
        elif abs(target_host - source_host) <= 2:
            return self.latency_profile.cxl_far_ns
        else:
            return self.latency_profile.cxl_network_ns
    
    def _handle_coherence_read(self, pool_id: int, addr: int, source_host: int) -> bool:
        """
        Handle cache coherence for read operations using MESI protocol
        
        Args:
            pool_id: Memory pool ID
            addr: Memory address
            source_host: Host performing read
            
        Returns:
            True if coherence miss occurred
        """
        with self.coherence_lock:
            key = (pool_id, addr)
            current_state = self.coherence_state.get(key, MESIState.INVALID)
            
            if current_state == MESIState.INVALID:
                # Read miss: fetch from memory
                self.coherence_state[key] = MESIState.EXCLUSIVE
                return True  # Coherence miss
            elif current_state in [MESIState.SHARED, MESIState.EXCLUSIVE, MESIState.MODIFIED]:
                # Read hit
                return False  # Coherence hit
            
        return False
    
    def _handle_coherence_write(self, pool_id: int, addr: int, source_host: int):
        """
        Handle cache coherence for write operations using MESI protocol
        
        Args:
            pool_id: Memory pool ID
            addr: Memory address  
            source_host: Host performing write
        """
        with self.coherence_lock:
            key = (pool_id, addr)
            current_state = self.coherence_state.get(key, MESIState.INVALID)

            # Invalidate other caches and gain exclusive access
            if current_state == MESIState.INVALID:
                # Write miss
                pass
            elif current_state == MESIState.SHARED:
                # Invalidate other sharers
                pass
            
            self.coherence_state[key] = MESIState.MODIFIED
            
            # In real implementation, would send invalidations to other hosts
    
    def simulate_network_contention(self, pool_id: int, base_contention_ns: int = 10):
        """
        Simulate network congestion and bandwidth contention
        
        Args:
            pool_id: The memory pool being accessed
            base_contention_ns: Base contention delay in nanoseconds
        """
        with self.pool_locks[pool_id]:
            num_accesses = self.concurrent_accesses[pool_id]
        
        if num_accesses > 1:
            # Add random delay to simulate contention
            delay_ns = random.randint(0, base_contention_ns * num_accesses)
            yield self.env.timeout(delay_ns)
    
    def inject_fault(self, fault_type: str = "transient", duration_ms: int = 100):
        """
        Inject faults for testing fault tolerance
        
        Args:
            fault_type: Type of fault ("transient", "host_failure", "network_partition")
            duration_ms: Fault duration in milliseconds
        """
        if fault_type == "transient":
            # Simulate transient memory error
            time.sleep(duration_ms / 1000)
            return False  # Indicate retry needed
        elif fault_type == "host_failure":
            # Simulate host failure by marking pools unavailable
            print(f"Simulating host failure for {duration_ms}ms")
            return False
        elif fault_type == "network_partition":
            # Simulate network partition
            print(f"Simulating network partition for {duration_ms}ms")
            return False
        
        return True
    
    def get_memory_layout(self) -> Dict[str, Any]:
        """
        Get current memory pool layout and usage
        
        Returns:
            Memory layout information
        """
        layout = {
            'pools': {},
            'total_allocated': 0,
            'fragmentation': 0.0
        }
        
        for pool_id, pool in self.memory_pools.items():
            pool_info = {
                'addresses': list(pool.keys()),
                'allocated_bytes': sum(len(data) for data in pool.values()),
                'num_regions': len(pool)
            }
            layout['pools'][pool_id] = pool_info
            layout['total_allocated'] += pool_info['allocated_bytes']
        
        return layout
    
    def benchmark_bandwidth(self, data_size_mb: int = 100, iterations: int = 10):
        """
        Benchmark memory bandwidth for different access patterns
        
        Args:
            data_size_mb: Size of test data in MB
            iterations: Number of benchmark iterations
            
        Returns:
            Bandwidth measurements
        """
        data_size = data_size_mb * 1024 * 1024
        test_data = np.random.bytes(data_size)
        
        results = {}
        
        # Test local access
        pool_id = 0
        addr = self.allocate_memory(pool_id, data_size)
        
        # Warm up
        for _ in range(2):
            yield self.env.process(self.write_memory(pool_id, addr, np.frombuffer(test_data, dtype=np.uint8), source_host=0))
            yield self.env.process(self.read_memory(pool_id, addr, data_size, source_host=0))
        
        # Benchmark local read/write
        start_time = self.env.now
        for _ in range(iterations):
            yield self.env.process(self.read_memory(pool_id, addr, data_size, source_host=0))
        local_read_time = (self.env.now - start_time) / iterations
        results['local_read_gbps'] = (data_size / (1024**3)) / (local_read_time / 1e9)
        
        # Benchmark remote access
        if len(self.memory_pools) > 1:
            remote_pool = 1
            remote_addr = self.allocate_memory(remote_pool, data_size)
            yield self.env.process(self.write_memory(remote_pool, remote_addr, np.frombuffer(test_data, dtype=np.uint8), source_host=1))
            
            start_time = self.env.now
            for _ in range(iterations):
                yield self.env.process(self.read_memory(remote_pool, remote_addr, data_size, source_host=0))
            remote_read_time = (self.env.now - start_time) / iterations
            results['remote_read_gbps'] = (data_size / (1024**3)) / (remote_read_time / 1e9)
        
        return results
    
    def run(self, process):
        self.env.run(until=self.env.process(process))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        stats = self.stats.copy()
        
        if self.stats['bandwidth_samples']:
            stats['avg_bandwidth_gbps'] = np.mean(self.stats['bandwidth_samples'])
            stats['peak_bandwidth_gbps'] = np.max(self.stats['bandwidth_samples'])
            
        if self.stats['latency_samples']:
            stats['avg_latency_ns'] = np.mean(self.stats['latency_samples'])
            stats['p99_latency_ns'] = np.percentile(self.stats['latency_samples'], 99)
        
        total_accesses = stats['local_accesses'] + stats['remote_accesses']
        if total_accesses > 0:
            stats['remote_access_ratio'] = stats['remote_accesses'] / total_accesses
            stats['coherence_miss_rate'] = stats['coherence_misses'] / total_accesses
        
        return stats
