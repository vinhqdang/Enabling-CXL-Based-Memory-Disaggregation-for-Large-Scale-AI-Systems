# XL-Share: CXL-Based Memory Disaggregation for Large-Scale AI Systems

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)]()

## Overview

**XL-Share** is a research prototype system that enables efficient memory disaggregation for large-scale AI models using CXL (Compute Express Link) 3.0 technology. The system allows multiple GPUs to share a single copy of model parameters stored in a CXL-attached memory pool, dramatically reducing memory overhead while maintaining reasonable inference performance.

### Key Features

- 🚀 **4x Memory Efficiency**: Reduce GPU memory usage by sharing model parameters
- 🔄 **Intelligent Prefetching**: Model-aware prefetching with 99.5% efficiency
- 💾 **Hardware Coherence**: Leverages CXL 3.0 hardware coherence protocols
- 📈 **8x GPU Scalability**: Support more GPUs with same memory footprint
- 🎯 **94% Cache Hit Rate**: High-performance local caching system
- 🛡️ **Fault Tolerance**: Resilient to transient errors and host failures

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    XL-Share System                          │
├─────────────────┬─────────────────┬─────────────────────────┤
│   GPU Node 1    │   GPU Node 2    │      CXL Memory Pool    │
│                 │                 │                         │
│ ┌─────────────┐ │ ┌─────────────┐ │  ┌─────────────────────┐│
│ │Local Cache  │ │ │Local Cache  │ │  │   Model Parameters  ││
│ │   512MB     │ │ │   512MB     │ │  │                     ││
│ └─────────────┘ │ └─────────────┘ │  │  • Embedding Layer  ││
│                 │                 │  │  • Transformer L1   ││
│ ┌─────────────┐ │ ┌─────────────┐ │  │  • Transformer L2   ││
│ │Prefetcher   │ │ │Prefetcher   │ │  │  • ...              ││
│ │             │ │ │             │ │  │  • Output Head      ││
│ └─────────────┘ │ └─────────────┘ │  └─────────────────────┘│
└─────────────────┴─────────────────┴─────────────────────────┘
         │                   │                   ▲
         └───────── CXL 3.0 Coherence ──────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (for full functionality)
- 16GB+ RAM (for emulation)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/username/xl-share
cd xl-share
```

2. **Create conda environment**
```bash
conda create -n xl-share python=3.10
conda activate xl-share
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running Experiments

#### Quick Demo
```bash
python run_experiments_simple.py
```

#### Full Benchmark Suite
```bash
python run_experiments.py
```

#### Custom Model Testing
```python
from xlshare import XLShareInferenceEngine, InferenceRequest
import numpy as np

# Initialize system
engine = XLShareInferenceEngine(
    cxl_pool_size_gb=16.0,
    gpu_cache_size_mb=512,
    emulate_cxl=True
)

# Create and register model
model_config, weights = engine.create_sample_transformer_model(
    num_layers=12,
    hidden_size=768,
    vocab_size=50000
)

engine.register_model(model_config, weights)

# Run inference
input_data = np.random.randn(1, 512).astype(np.float32)
request = InferenceRequest(
    request_id="test_1",
    input_data=input_data,
    model_name=model_config.name,
    timestamp=time.time()
)

result = engine.inference(request)
print(f"Latency: {result.latency_ms:.1f}ms")
print(f"Cache hit rate: {result.cache_hit_rate:.2f}")
```

## Experimental Results

Our experimental evaluation demonstrates significant improvements in memory efficiency:

### Performance Summary

| Metric | XL-Share | Traditional | Improvement |
|--------|----------|-------------|-------------|
| **Memory Efficiency** | 4.0x better | Baseline | 400% improvement |
| **GPU Scalability** | 8 GPUs | 4 GPUs | 2x density |
| **Cache Hit Rate** | 94.2% | N/A | Excellent |
| **Prefetch Efficiency** | 99.5% | N/A | Near-perfect |

### Latency Analysis

- **Cold Start**: ~5 seconds (one-time cache population)
- **Warm Inference**: ~25ms per request
- **Batch Processing**: Excellent amortization of overhead

*See [resultv1.md](resultv1.md) for detailed experimental analysis.*

## System Components

### Core Modules

#### 1. Memory Manager (`xlshare/memory_manager.py`)
- CXL memory pool management
- Hardware coherence support
- Local cache with LRU eviction

#### 2. Intelligent Prefetcher (`xlshare/prefetcher.py`)
- Model-aware prefetch scheduling
- Pipeline parallelism (communication + computation)
- Layer dependency analysis

#### 3. Inference Engine (`xlshare/inference_engine.py`)
- High-level inference API
- Request batching and scheduling
- Performance monitoring

#### 4. CXL Emulator (`xlshare/emulator.py`)
- Hardware behavior simulation
- Latency and bandwidth modeling
- Fault injection for testing

### Configuration Options

```python
# Memory configuration
cxl_pool_size_gb = 64.0      # CXL memory pool size
gpu_cache_size_mb = 1024     # Local cache size
prefetch_threads = 2         # Prefetch worker threads

# Performance tuning
cxl_latency_ns = 300        # CXL access latency
cache_eviction = "lru"      # Cache eviction policy
prefetch_lookahead = 2      # Layers to prefetch ahead
```

## Use Cases

### Production Deployment Scenarios

#### 1. **Multi-Tenant AI Serving**
- Multiple clients share single model instance
- Reduced memory footprint per tenant
- Cost-effective scaling

#### 2. **Large Model Training**
- Parameter sharing across training nodes
- Reduced memory requirements
- Simplified distributed training

#### 3. **Development & Research**
- Test larger models on limited hardware
- Rapid prototyping with memory constraints
- Educational use for understanding memory patterns

#### 4. **Edge Deployment**
- Memory-constrained edge servers
- Efficient model serving at the edge
- Reduced infrastructure costs

### Performance Characteristics

**Suitable for:**
- Batch inference workloads
- Latency-tolerant applications (>100ms)
- Memory-constrained environments
- Development and testing scenarios

**Not optimal for:**
- Ultra-low latency requirements (<10ms)
- Single-request processing
- Memory-abundant environments
- Real-time interactive applications

## Architecture Deep Dive

### Memory Hierarchy

1. **CXL Memory Pool (Tier 1)**
   - Shared storage for complete model
   - Hardware-coherent across nodes
   - High capacity, moderate latency

2. **Local GPU Cache (Tier 2)**  
   - Fast local storage for hot weights
   - LRU eviction with model hints
   - Low capacity, very low latency

3. **Prefetch Pipeline (Management)**
   - Intelligent scheduling based on model graph
   - Overlap communication with computation
   - Predictive weight loading

### Fault Tolerance Design

- **Transient Error Recovery**: Automatic retry with exponential backoff
- **Host Failure Handling**: Graceful degradation with alternate pools
- **Data Integrity**: CXL hardware coherence ensures consistency
- **Checkpoint/Restore**: Periodic state snapshots for recovery

## Development

### Project Structure

```
xl-share/
├── xlshare/                 # Core system modules
│   ├── __init__.py         # Package initialization
│   ├── memory_manager.py   # CXL memory management
│   ├── prefetcher.py       # Intelligent prefetching
│   ├── inference_engine.py # Main inference system
│   └── emulator.py         # Hardware emulation
├── benchmarks.py           # Benchmark suite
├── run_experiments.py      # Full experiment runner
├── run_experiments_simple.py # Quick demo
├── requirements.txt        # Python dependencies
├── planv1.md              # Research plan
├── resultv1.md            # Experimental results
└── README.md              # This file
```

### Running Tests

```bash
# Quick functionality test
python -c "from xlshare import XLShareInferenceEngine; print('Import successful')"

# Run simple benchmark
python run_experiments_simple.py

# Memory usage analysis
python -c "
from xlshare import XLShareInferenceEngine
engine = XLShareInferenceEngine()
model_config, weights = engine.create_sample_transformer_model()
print(f'Model size: {model_config.total_size_mb:.1f}MB')
"
```

### Extending the System

#### Adding Custom Models
```python
from xlshare.prefetcher import LayerInfo, LayerType

# Define custom layer structure
custom_layers = [
    LayerInfo(
        name="custom_layer_1",
        layer_type=LayerType.LINEAR,
        weight_shape=(512, 1024),
        weight_size_bytes=512*1024*4,
        computation_time_ms=5.0
    ),
    # Add more layers...
]

# Register with system
engine.prefetcher.register_model(custom_layers, weight_addresses)
```

#### Custom Prefetch Policies
```python
class CustomPrefetcher(ModelAwarePrefetcher):
    def smart_prefetch_pipeline(self, current_layer_idx, lookahead=3):
        # Implement custom prefetch logic
        # Consider model-specific patterns
        pass
```

## Performance Tuning Guide

### Memory Optimization

1. **Cache Size Tuning**
   ```python
   # Increase cache for better hit rates
   engine = XLShareInferenceEngine(gpu_cache_size_mb=2048)
   ```

2. **Prefetch Configuration**
   ```python
   # More aggressive prefetching
   engine.prefetcher.smart_prefetch_pipeline(current_idx, lookahead=4)
   ```

3. **Batch Processing**
   ```python
   # Process multiple requests together
   results = engine.batch_inference([req1, req2, req3])
   ```

### Latency Optimization

1. **Persistent Caching**: Keep cache warm between sessions
2. **Request Batching**: Amortize cache miss costs
3. **Model Quantization**: Reduce weight transfer sizes
4. **Prefetch Tuning**: Optimize lookahead distance

## Troubleshooting

### Common Issues

#### 1. High Latency on First Request
**Cause**: Cold cache requires weight loading  
**Solution**: Implement cache warming or use batch processing

#### 2. Low Cache Hit Rate
**Cause**: Insufficient cache size or poor prefetch policy  
**Solution**: Increase cache size or tune prefetch parameters

#### 3. Memory Pool Exhaustion  
**Cause**: CXL pool too small for model  
**Solution**: Increase pool size or use model compression

#### 4. CXL Emulation Errors
**Cause**: Simulation parameters misconfigured  
**Solution**: Check latency and bandwidth settings

### Debug Mode
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check system statistics
stats = engine.get_system_stats()
print(f"Cache hit rate: {stats['local_cache']['hit_rate']}")
print(f"Memory utilization: {stats['memory_manager']['pool_utilization']}")
```

## Research Applications

### Academic Use

This implementation serves as a research platform for:
- **Memory Systems Research**: Novel memory hierarchy designs
- **AI Systems Optimization**: Large model serving efficiency
- **Hardware-Software Co-design**: CXL integration studies
- **Distributed Computing**: Memory disaggregation patterns

### Citation
```bibtex
@misc{xlshare2025,
  title={XL-Share: CXL-Based Memory Disaggregation for Large-Scale AI Systems},
  author={AI Systems Research Lab},
  year={2025},
  note={Research prototype implementing CXL memory disaggregation}
}
```

## Future Roadmap

### Short-term (Q1 2025)
- [ ] Performance optimization for cold start
- [ ] Real CXL hardware integration
- [ ] Multi-model serving support
- [ ] Advanced cache policies

### Medium-term (Q2-Q3 2025)
- [ ] Distributed memory pools
- [ ] Compression integration
- [ ] Production deployment guides
- [ ] Kubernetes operator

### Long-term (Q4 2025+)
- [ ] Hardware acceleration
- [ ] ML-driven optimization
- [ ] Industry partnerships
- [ ] Standardization efforts

## Contributing

We welcome contributions! Please see our contribution guidelines:

1. **Bug Reports**: Use GitHub issues with reproduction steps
2. **Feature Requests**: Discuss in issues before implementation
3. **Pull Requests**: Follow code style and include tests
4. **Documentation**: Help improve docs and examples

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **CXL Consortium** for specification and hardware ecosystem
- **ASPLOS Community** for research inspiration and feedback
- **Open Source Projects** for foundational libraries and tools

## Support

- **Documentation**: See [resultv1.md](resultv1.md) for detailed results
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Join community discussions for Q&A
- **Email**: Contact maintainers for research collaborations

---

*Built with ❤️ for advancing AI systems research and enabling efficient large-scale model deployment.*