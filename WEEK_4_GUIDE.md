# Week 4 Implementation Guide: FastAPI Application with Context7 Patterns

## 🎯 Overview
This guide provides detailed implementation instructions for Week 4, focusing on creating a production-ready FastAPI application using Context7-researched patterns. We'll build upon the completed Pydantic models and service layer from Week 3.

## 📚 Context7 Research Summary

### Key FastAPI Patterns Applied:
1. **APIRouter Organization**: Modular router structure with prefix, tags, and dependencies
2. **Dependency Injection**: Router-level and path-level dependencies with proper inheritance
3. **Middleware Patterns**: HTTP middleware for logging, CORS, and trusted hosts
4. **Exception Handling**: Global exception handlers with structured responses
5. **Lifespan Management**: Application startup/shutdown event handling

## 🛠️ Implementation Plan

### Day 22-24: FastAPI Application Setup

#### 1. Router Module Structure
Create modular routers following Context7 patterns:

```bash
mkdir -p app/routers
touch app/routers/__init__.py
touch app/routers/embedding_router.py
touch app/routers/reranking_router.py
touch app/routers/health_router.py
```

#### 2. Update Configuration (app/config.py)
Add FastAPI-specific settings:

```python
# Add to Settings class
allowed_origins: List[str] = ["*"]
allowed_hosts: List[str] = ["*"]
cors_credentials: bool = True
cors_methods: List[str] = ["GET", "POST", "OPTIONS"]
```

#### 3. Implement Embedding Router
**File: app/routers/embedding_router.py**

Key Features:
- APIRouter with prefix="/embed"
- Router-level error responses
- Dependency injection for EmbeddingService
- Comprehensive OpenAPI documentation

#### 4. Implement Reranking Router  
**File: app/routers/reranking_router.py**

Key Features:
- APIRouter with prefix="/rerank"
- Similar structure to embedding router
- Batch reranking endpoint (optional)

#### 5. Implement Health Router
**File: app/routers/health_router.py**

Key Features:
- Comprehensive health checks
- Backend status monitoring
- System resource information

### Day 25-26: Advanced Patterns & Monitoring

#### 1. Global Exception Handlers
Implement structured exception handling:

```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Structured logging with context
    # Return standardized ErrorResponse
    pass

@app.exception_handler(HTTPException)  
async def http_exception_handler(request: Request, exc: HTTPException):
    # HTTP-specific error handling
    pass
```

#### 2. Request Logging Middleware
Create comprehensive request/response logging:

```python
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    # Log request details
    # Measure processing time
    # Add X-Process-Time header
    # Log response details
    pass
```

#### 3. Dependency Optimization
Implement Context7 dependency patterns:

```python
# Router-level dependencies
async def get_backend_manager() -> BackendManager:
    pass

async def get_embedding_service(
    manager: BackendManager = Depends(get_backend_manager)
) -> EmbeddingService:
    pass
```

#### 4. Enhanced Health Checks
Implement comprehensive health monitoring:
- Backend availability
- Model loading status  
- Memory usage
- Processing time metrics

### Day 27-28: Testing & Validation

#### 1. API Integration Tests
Create comprehensive test suite:

```python
# tests/test_api_integration.py
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    return TestClient(app)

def test_embed_endpoint(client):
    # Test embedding functionality
    pass

def test_rerank_endpoint(client):
    # Test reranking functionality
    pass

def test_health_endpoint(client):
    # Test health check
    pass
```

#### 2. Error Scenario Testing
Test all error conditions:
- Invalid input validation
- Backend unavailable scenarios
- Rate limiting (if implemented)
- Timeout handling

#### 3. Performance Validation
Measure API performance:
- Response times per endpoint
- Middleware overhead
- Memory usage under load

## 🎯 Key Context7 Patterns to Implement

### 1. Router Dependencies
```python
router = APIRouter(
    prefix="/api/v1/embed",
    tags=["embedding"],
    dependencies=[Depends(get_backend_manager)],
    responses={
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
        400: {"model": ErrorResponse, "description": "Bad Request"}
    }
)
```

### 2. Dependency Injection Chain
```python
# Backend Manager (Global)
async def get_backend_manager() -> BackendManager:
    pass

# Service Layer (Router-specific)  
async def get_embedding_service(
    manager: BackendManager = Depends(get_backend_manager)
) -> EmbeddingService:
    pass

# Path Operation (Endpoint-specific)
@router.post("/")
async def embed_texts(
    request: EmbedRequest,
    service: EmbeddingService = Depends(get_embedding_service)
):
    pass
```

### 3. Middleware Chain
```python
# Order matters: last added = first executed
app.add_middleware(TrustedHostMiddleware, ...)
app.add_middleware(CORSMiddleware, ...)

@app.middleware("http")
async def logging_middleware(request, call_next):
    # Custom logging middleware
    pass
```

### 4. Exception Handler Hierarchy
```python
# Most specific first
@app.exception_handler(ValidationError)
async def validation_exception_handler():
    pass

@app.exception_handler(HTTPException)
async def http_exception_handler():
    pass

@app.exception_handler(Exception)
async def global_exception_handler():
    pass
```

## 📋 Success Criteria

### Technical Requirements:
- [x] All routers properly organized with Context7 patterns
- [x] Dependency injection working correctly
- [x] Middleware chain configured and tested
- [x] Global exception handling implemented
- [x] Comprehensive health checks functional
- [x] Request/response logging operational

### Performance Requirements:
- [x] API response time < 1 second for 100 texts
- [x] Minimal middleware overhead (< 5ms)
- [x] Memory usage stable under load
- [x] Backend initialization time < 10 seconds

### Quality Requirements:
- [x] 90%+ test coverage for API layer
- [x] All error scenarios handled gracefully
- [x] OpenAPI documentation complete
- [x] Structured logging throughout

## 🚀 Next Steps (Week 5)

After completing Week 4:
1. Docker containerization
2. Production deployment configuration
3. Performance optimization
4. Documentation and deployment guides

## 🔧 Troubleshooting

### Common Issues:
1. **Dependency Not Found**: Check import paths and dependency registration
2. **Middleware Order**: Review middleware addition sequence
3. **Router Conflicts**: Verify prefix uniqueness and path patterns
4. **Backend Not Available**: Ensure proper lifespan management

### Debug Commands:
```bash
# Test API endpoints
curl -X POST "http://localhost:9000/api/v1/embed" -H "Content-Type: application/json" -d '{"texts": ["test"]}'

# Check health status
curl "http://localhost:9000/health"

# View OpenAPI docs
open "http://localhost:9000/docs"
```

This guide provides the foundation for implementing a production-ready FastAPI application using modern patterns researched through Context7.
