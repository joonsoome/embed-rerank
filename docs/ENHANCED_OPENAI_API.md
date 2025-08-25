# 🚀 Enhanced OpenAI Compatibility with Apple MLX Arguments

This document describes the enhanced OpenAI-compatible API that provides full OpenAI SDK compatibility while offering additional configurable arguments for Apple MLX optimization.

## ✨ Key Features

- **🔄 Perfect OpenAI Compatibility**: Existing OpenAI SDK code works unchanged
- **🔧 Enhanced Configuration**: Optional MLX-specific parameters for optimization
- **🌐 Multiple Configuration Methods**: Request body fields, custom headers, or query parameters
- **⚡ Apple Silicon Optimization**: Full control over MLX backend behavior
- **📊 Enhanced Metrics**: Optional detailed performance information

## 🎯 Configuration Options

### Request Body Arguments (Recommended)

Add these optional fields to your OpenAI embeddings request:

```json
{
  "input": ["Your text here"],
  "model": "text-embedding-ada-002",
  
  // 🚀 Enhanced MLX Configuration (All Optional)
  "batch_size": 32,                    // 📦 Processing batch size (1-128)
  "normalize": true,                   // 🎯 Normalize embeddings to unit length
  "backend_preference": "mlx",         // 🧠 Backend: "mlx", "torch", "auto"
  "device_preference": "mps",          // ⚡ Device: "mps", "cpu", "auto"  
  "max_tokens_per_text": 512,         // 📏 Maximum tokens per input text
  "return_timing": false               // ⏱️ Include detailed timing in response
}
```

### Custom Headers (Enterprise Integration)

Alternative configuration via HTTP headers:

```bash
curl -X POST http://localhost:9000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "X-MLX-Batch-Size: 64" \
  -H "X-MLX-Normalize: true" \
  -H "X-MLX-Backend: mlx" \
  -H "X-MLX-Device: mps" \
  -d '{"input": ["Hello MLX"], "model": "text-embedding-ada-002"}'
```

## 📊 Enhanced Response Format

When `return_timing: true`, the usage object includes additional metrics:

```json
{
  "object": "list",
  "data": [...],
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10,
    
    // 🚀 Enhanced MLX Metrics (when return_timing=true)
    "mlx_processing_time": 0.0045,     // ⚡ MLX inference time (seconds)
    "total_processing_time": 0.0123,   // 🕐 Total request time (seconds)
    "backend_used": "MLXBackend",       // 🧠 Actual backend used
    "device_used": "mps",               // 💻 Device used for processing
    "batch_size_used": 32               // 📦 Actual batch size used
  }
}
```

## 🔧 Configuration Parameter Details

### `batch_size` (integer, 1-128)
- **Default**: Auto-calculated based on input size
- **Purpose**: Controls how many texts are processed together
- **Optimization**: Larger batches = better throughput, smaller batches = lower latency
- **Recommended**: 32 for balanced performance, 64+ for high throughput

### `normalize` (boolean)
- **Default**: `true` (OpenAI compatibility)
- **Purpose**: Whether to normalize embeddings to unit length
- **Use Cases**: 
  - `true`: For cosine similarity (most common)
  - `false`: For applications needing raw embedding magnitudes

### `backend_preference` (string)
- **Options**: `"mlx"`, `"torch"`, `"auto"`
- **Default**: Uses server configuration
- **Purpose**: Force specific backend usage
- **Use Cases**:
  - `"mlx"`: Maximum Apple Silicon performance
  - `"torch"`: PyTorch MPS fallback
  - `"auto"`: Automatic selection based on hardware

### `device_preference` (string)
- **Options**: `"mps"`, `"cpu"`, `"auto"`
- **Default**: Uses backend's device detection
- **Purpose**: Control which device processes the request
- **Use Cases**:
  - `"mps"`: Apple Silicon acceleration
  - `"cpu"`: CPU processing (compatibility)
  - `"auto"`: Best available device

### `max_tokens_per_text` (integer, 1-8192)
- **Default**: Model's maximum context length
- **Purpose**: Truncate input texts to specified token limit
- **Use Cases**: Memory optimization, consistent processing times

### `return_timing` (boolean)
- **Default**: `false`
- **Purpose**: Include detailed performance metrics in response
- **Use Cases**: Performance monitoring, optimization, debugging

## 💡 Usage Examples

### Basic OpenAI Compatibility
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9000/v1", api_key="dummy")

response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=["Hello Apple MLX!"]
)
```

### Enhanced Configuration
```python
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=["Hello Apple MLX!", "Fast embeddings"],
    extra_body={
        "batch_size": 16,
        "normalize": True,
        "backend_preference": "mlx",
        "return_timing": True
    }
)

# Access enhanced metrics
if hasattr(response.usage, 'mlx_processing_time'):
    print(f"MLX processing: {response.usage.mlx_processing_time}s")
```

### High-Performance Configuration
```python
# Optimized for maximum throughput
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=batch_of_texts,
    extra_body={
        "batch_size": 64,           # Large batches
        "backend_preference": "mlx", # Force MLX
        "device_preference": "mps",  # Force Apple Silicon
        "return_timing": True        # Monitor performance
    }
)
```

### Low-Latency Configuration
```python
# Optimized for minimum latency
response = client.embeddings.create(
    model="text-embedding-ada-002", 
    input=["Single text for fast processing"],
    extra_body={
        "batch_size": 1,            # Minimal batching
        "backend_preference": "mlx",
        "return_timing": True
    }
)
```

## 🎯 Best Practices

### Batch Size Optimization
- **Small datasets (1-10 texts)**: `batch_size: 1-8`
- **Medium datasets (10-100 texts)**: `batch_size: 16-32`  
- **Large datasets (100+ texts)**: `batch_size: 32-64`
- **High-throughput applications**: `batch_size: 64-128`

### Backend Selection
- **Production workloads**: `backend_preference: "mlx"` for maximum performance
- **Development/testing**: `backend_preference: "auto"` for flexibility
- **Compatibility testing**: `backend_preference: "torch"` for fallback validation

### Performance Monitoring
- Always use `return_timing: true` during development and tuning
- Monitor `mlx_processing_time` vs `total_processing_time` to identify bottlenecks
- Track `backend_used` and `device_used` to verify configuration

### Error Handling
```python
try:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts,
        extra_body={"batch_size": 64, "return_timing": True}
    )
except Exception as e:
    # Handle configuration errors gracefully
    # Fallback to basic configuration
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
```

## 🔍 Troubleshooting

### Common Issues

1. **Large batch sizes causing memory errors**
   - Reduce `batch_size` to 16-32
   - Set `max_tokens_per_text` to limit input size

2. **Slower than expected performance**
   - Ensure `backend_preference: "mlx"`
   - Verify `device_used: "mps"` in response
   - Increase `batch_size` for throughput

3. **Configuration not taking effect**
   - Check you're using `extra_body` parameter with OpenAI SDK
   - Verify custom headers are properly formatted
   - Enable `return_timing` to see actual settings used

### Performance Expectations

| Configuration | Expected Latency | Throughput |
|---------------|-----------------|------------|
| Small batch (1-8) | < 10ms | 100-500 texts/sec |
| Medium batch (16-32) | 20-50ms | 500-1000 texts/sec |
| Large batch (64+) | 50-100ms | 1000+ texts/sec |

*Performance varies based on text length, hardware, and model size*

## 🎨 Integration Examples

### Streamlit App
```python
import streamlit as st
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9000/v1", api_key="dummy")

# UI controls for MLX configuration
batch_size = st.slider("Batch Size", 1, 64, 32)
normalize = st.checkbox("Normalize Embeddings", True)
show_timing = st.checkbox("Show Performance Metrics", False)

if st.button("Generate Embeddings"):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=st.text_area("Enter texts").split('\n'),
        extra_body={
            "batch_size": batch_size,
            "normalize": normalize,
            "return_timing": show_timing
        }
    )
    
    st.write(f"Generated {len(response.data)} embeddings")
    if show_timing and hasattr(response.usage, 'mlx_processing_time'):
        st.metric("MLX Processing Time", f"{response.usage.mlx_processing_time:.4f}s")
```

### FastAPI Integration
```python
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
client = OpenAI(base_url="http://localhost:9000/v1", api_key="dummy")

@app.post("/embed")
async def embed_texts(texts: List[str], batch_size: int = 32):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts,
        extra_body={
            "batch_size": batch_size,
            "backend_preference": "mlx",
            "return_timing": True
        }
    )
    
    return {
        "embeddings": [d.embedding for d in response.data],
        "performance": {
            "mlx_time": response.usage.mlx_processing_time,
            "total_time": response.usage.total_processing_time,
            "backend": response.usage.backend_used
        }
    }
```

## 🚀 Next Steps

1. **Run the examples**: Use `examples/enhanced_openai_usage.py` to test all features
2. **Performance tuning**: Experiment with different batch sizes for your use case
3. **Monitor metrics**: Use `return_timing: true` to optimize configuration
4. **Enterprise integration**: Implement custom headers for centralized configuration

---

**🎯 You now have a fully configurable OpenAI-compatible endpoint that gives you complete control over Apple MLX performance while maintaining perfect SDK compatibility!**
