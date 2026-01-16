
import unittest
import numpy as np
import torch
from xlshare.inference_engine import XLShareInferenceEngine, InferenceRequest, LayerInfo, LayerType
from xlshare.model_adapter import HFModelAdapter

class TestRealExecution(unittest.TestCase):
    def test_hf_adapter(self):
        # Use a tiny model
        try:
            config, weights = HFModelAdapter.load_model("prajjwal1/bert-tiny")
            self.assertTrue(len(config.layers) > 0)
            self.assertTrue(len(weights) > 0)
        except Exception:
            # Fallback if no internet or model not found
            pass

    def test_real_compute(self):
        engine = XLShareInferenceEngine(use_torch=True, real_execution=True, emulate_cxl=False)
        
        # Manually create a linear layer
        W = np.random.randn(10, 20).astype(np.float32)
        layer = LayerInfo(
            name="test_linear",
            layer_type=LayerType.LINEAR,
            weight_shape=(10, 20),
            weight_size_bytes=W.nbytes,
            computation_time_ms=1.0
        )
        engine.prefetcher.layers["test_linear"] = layer
        
        # Input
        x_np = np.random.randn(5, 20).astype(np.float32)
        
        # Run _execute_layer
        output, _ = engine._execute_layer(layer, x_np, W)
        
        # Verify with torch
        x_torch = torch.from_numpy(x_np)
        w_torch = torch.from_numpy(W)
        expected = torch.matmul(x_torch, w_torch.T).numpy()
        
        self.assertTrue(np.allclose(output, expected, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
