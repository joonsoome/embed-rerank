# 🔧 Troubleshooting Guide

이 문서는 Apple MLX Embed-Rerank API 사용 중 발생할 수 있는 문제와 해결 방법을 제공합니다.

## 🚨 Common Issues & Solutions

### 1. "Embedding service not initialized" Error

**문제 증상:**
```
RuntimeError: Embedding service not initialized. Server startup may have failed.
HTTP 500 Internal Server Error
```

**원인:**
OpenAI 및 TEI 호환성 라우터에서 embedding service가 제대로 초기화되지 않았을 때 발생합니다.

**해결 방법:**

#### v1.2.0 이후 (자동 해결됨):
이 문제는 v1.2.0에서 완전히 해결되었습니다. PyPI 패키지를 업데이트하세요:

```bash
pip install --upgrade embed-rerank
embed-rerank
```

#### 소스코드 사용자 (수동 해결):

1. **main.py에서 embedding service 설정 확인:**

```python
# app/main.py의 lifespan 함수에서
# 🎯 Initialize embedding service for OpenAI and TEI compatibility
from .services.embedding_service import EmbeddingService
embedding_service = EmbeddingService(backend_manager)

# 🔗 Set embedding service for OpenAI and TEI routers
openai_router.set_embedding_service(embedding_service)
tei_router.set_embedding_service(embedding_service)
```

2. **서버 재시작:**

```bash
# 기존 서버 종료
pkill -f "uvicorn.*embed-rerank"

# 서버 재시작
./tools/server-run.sh
# 또는
python -m uvicorn app.main:app --host 0.0.0.0 --port 9000
```

3. **초기화 로그 확인:**
서버 시작 시 다음 로그가 출력되어야 합니다:
```
🔄 OpenAI router updated with dynamic embedding service
🔄 TEI router updated with dynamic embedding service  
```

4. **테스트 실행:**
```bash
# OpenAI API 테스트
curl -X POST "http://localhost:9000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input": ["test"], "model": "text-embedding-ada-002"}'

# TEI API 테스트  
curl -X POST "http://localhost:9000/embed" \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["test"]}'
```

### 2. MLX reranker 로딩 실패 (quant_method 오류)

**문제 증상 (로그 예시):**
```
Failed to load CrossEncoder model 'vserifsaglam/Qwen3-Reranker-4B-4bit-MLX':
The model's quantization config ... has no `quant_method` attribute
```

**원인:**
- `RERANKER_BACKEND=auto`는 **reranker에서 Torch를 기본 선택**합니다.
- MLX 전용 모델(`...-MLX`)을 Torch CrossEncoder로 로딩하면 실패합니다.

**해결 방법:**
1. **MLX reranker 사용 (Apple Silicon):**
```bash
# .env
RERANKER_BACKEND=mlx
RERANKER_MODEL_NAME=vserifsaglam/Qwen3-Reranker-4B-4bit-MLX

# (선택) 기본 HF 캐시에 미리 다운로드
./download-rerank-model.sh
```
2. **Torch reranker 유지:**
```bash
RERANKER_BACKEND=torch
RERANKER_MODEL_ID=cross-encoder/ms-marco-MiniLM-L-6-v2
```
3. **서비스 재시작 후 /health 또는 /api/v1/rerank 확인**

### 3. API 호환성 테스트 실패

**문제 증상:**
```bash
./tools/server-tests.sh --api-compatibility
# 일부 API 테스트 실패
```

**해결 방법:**

1. **개별 API 테스트:**
```bash
# Native API 테스트
TEST_SERVER_URL="http://localhost:9000" python -m pytest tests/test_native_api.py -v

# OpenAI API 테스트  
TEST_SERVER_URL="http://localhost:9000" python -m pytest tests/test_openai_api.py -v

# TEI API 테스트
TEST_SERVER_URL="http://localhost:9000" python -m pytest tests/test_tei_api.py -v

# Cohere API 테스트
TEST_SERVER_URL="http://localhost:9000" python -m pytest tests/test_cohere_api.py -v
```

2. **서버 상태 확인:**
```bash
curl http://localhost:9000/health/
```

3. **로그 확인:**
```bash
tail -f server.log  # 백그라운드 서버인 경우
```

### 4. 포트 충돌 문제

**문제 증상:**
```
OSError: [Errno 48] Address already in use
```

**해결 방법:**

1. **사용 중인 포트 확인:**
```bash
lsof -i :9000
```

2. **프로세스 종료:**
```bash
# 특정 PID 종료
kill <PID>

# 모든 embed-rerank 프로세스 종료
pkill -f "uvicorn.*embed-rerank"
```

3. **다른 포트 사용:**
```bash
embed-rerank --port 8080
# 또는
python -m uvicorn app.main:app --port 8080
```

### 5. 모델 로딩 실패

**문제 증상:**
```
Failed to load model: mlx-community/Qwen3-Embedding-4B-4bit-DWQ
```

**해결 방법:**

1. **인터넷 연결 확인:**
모델 첫 다운로드 시 인터넷이 필요합니다.

2. **캐시 확인:**
```bash
ls ~/.cache/huggingface/hub/ | grep -i qwen3
```

3. **수동 모델 다운로드:**
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("mlx-community/Qwen3-Embedding-4B-4bit-DWQ")
# 모델이 자동으로 캐시됩니다
```

4. **디스크 공간 확인:**
모델은 약 2.3GB의 공간이 필요합니다.

### 6. Apple Silicon이 아닌 환경

**문제 증상:**
```
MLX not available, falling back to PyTorch
```

**해결 방법:**

이는 정상적인 동작입니다. Intel Mac이나 다른 플랫폼에서는 자동으로 PyTorch 백엔드로 전환됩니다:

1. **백엔드 확인:**
```bash
curl http://localhost:9000/health/
# "backend": "TorchBackend" 출력되면 정상
```

2. **성능 기대치 조정:**
- Apple Silicon (MLX): < 1ms
- Intel Mac (PyTorch MPS): 10-50ms  
- Other (PyTorch CPU): 100-500ms

### 7. 가상환경 문제

**문제 증상:**
```
ModuleNotFoundError: No module named 'app'
```

**해결 방법:**

1. **가상환경 활성화 확인:**
```bash
echo $VIRTUAL_ENV
# .venv 경로가 출력되어야 함
```

2. **가상환경 활성화:**
```bash
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

3. **의존성 설치:**
```bash
pip install -r requirements.txt
```

## 🔍 Debugging Tips

### 로그 레벨 조정
```bash
embed-rerank --log-level DEBUG
```

### 상세한 에러 정보
```bash
# pytest에서 상세 정보
python -m pytest tests/ -v --tb=long

# 서버에서 스택 트레이스 활성화
export DEBUG=1
embed-rerank
```

### Health Check 활용
```bash
# 전체 시스템 상태 확인
curl http://localhost:9000/health/ | jq '.'

# 특정 API 상태 확인
curl http://localhost:9000/v1/health    # OpenAI 호환성
curl http://localhost:9000/info         # TEI 호환성
```

### 성능 모니터링
```bash
# 내장 성능 테스트
embed-rerank --test performance --test-url http://localhost:9000

# 실시간 모니터링
curl -H "X-MLX-Return-Timing: true" \
  -X POST "http://localhost:9000/v1/embeddings" \
  -d '{"input": ["test"], "return_timing": true}'
```

## 🆘 Getting Help

1. **GitHub Issues**: [Report bugs or request features](https://github.com/joonsoo-me/embed-rerank/issues)
2. **Discussions**: [Community support and questions](https://github.com/joonsoo-me/embed-rerank/discussions)
3. **Documentation**: [Complete API documentation](http://localhost:9000/docs)

## 📋 버전별 변경사항

### v1.2.0 (최신)
- ✅ **해결됨**: "Embedding service not initialized" 에러
- 🆕 **추가됨**: Cohere API v1/v2 호환성
- 🆕 **추가됨**: 4개 API 동시 지원 (Native, OpenAI, TEI, Cohere)
- 🔧 **개선됨**: 자동 embedding service 초기화

### v1.1.0
- 🆕 TEI API 호환성 추가
- 🔧 성능 최적화

### v1.0.0
- 🚀 초기 릴리스
- ✅ Native API 및 OpenAI 호환성
