
import unittest
import numpy as np
from xlshare.memory_manager import LocalCache

class TestEviction(unittest.TestCase):
    def test_lru_behavior(self):
        # Cache size 2 items (assume 1MB each for simplicity in thinking, but code uses MB limit)
        # 1 item = 1MB. Cap = 2MB.
        cache = LocalCache(capacity_mb=2)
        
        data = np.zeros(1024*1024, dtype=np.uint8) # 1MB
        
        cache.put("A", data)
        cache.put("B", data) # Cache: A, B (MRU)
        
        # Access A
        cache.get("A") # Cache: B, A (MRU)
        
        # Add C, should evict B (LRU)
        cache.put("C", data)
        
        self.assertIsNotNone(cache.get("A"))
        self.assertIsNotNone(cache.get("C"))
        self.assertIsNone(cache.get("B"))

    def test_graph_aware_behavior(self):
        cache = LocalCache(capacity_mb=2)
        cache.set_eviction_policy("graph_aware")
        
        data = np.zeros(1024*1024, dtype=np.uint8) # 1MB
        
        cache.put("A", data)
        cache.put("B", data) 
        
        # Future: A needed at 10, B needed at 100.
        # Even if we access A now (making it MRU), B is further away.
        # Wait, if we access A, it becomes MRU. 
        # LRU would evict B if we didn't touch it.
        # Let's make A the LRU implies "A accessed long ago", but "A needed soon".
        
        # Access B to make it MRU
        cache.get("B") # Cache: A (LRU), B (MRU)
        
        # Future: A needed at t=5. B needed at t=50.
        cache.update_future_accesses({"A": 5, "B": 50})
        
        # Add C using 1MB.
        # LRU would evict A.
        # Graph-Aware should evict B (furthest away).
        cache.put("C", data)
        
        self.assertIsNotNone(cache.get("A"), "A should remain because it is needed soon")
        self.assertIsNone(cache.get("B"), "B should be evicted because it is needed late")
        self.assertIsNotNone(cache.get("C"))

if __name__ == '__main__':
    unittest.main()
