## 개발자 매뉴얼: LangChain 기반 네이버 검색 RAG 시스템 구축

본 문서는 LangChain, Ollama (Llama 3), 그리고 네이버 검색 API를 활용하여 주기적으로 데이터를 갱신하고, Llama 3가 저장된 데이터를 기반으로 답변하는 RAG(Retrieval-Augmented Generation) 시스템을 구축하는 방법에 대한 개발자 매뉴얼입니다.

---

### 1. 개요 (Overview)

이 시스템은 다음과 같은 목표를 가집니다:

- **지식 절단점 극복**: Llama 3와 같은 대규모 언어 모델(LLM)의 고정된 학습 데이터(Knowledge Cut-off) 한계를 넘어 최신 정보에 접근합니다.
- **환각(Hallucination) 감소**: LLM이 검색된 실제 정보를 기반으로 답변을 생성하여 잘못된 정보 생성을 줄입니다.
- **주기적 데이터 갱신**: 네이버 검색 API를 통해 최신 정보를 주기적으로 수집하여 RAG 데이터 소스를 업데이트합니다.

**핵심 기술 스택:**

- **언어 모델**: Ollama (Llama 3)
- **RAG 프레임워크**: LangChain
- **외부 지식 소스**: 네이버 검색 API
- **벡터 데이터베이스**: ChromaDB (로컬 환경 기준)
- **임베딩 모델**: Ollama (nomic-embed-text)

---

### 2. 선수 조건 (Prerequisites)

시스템을 구축하기 전에 다음 소프트웨어 및 라이브러리가 설치되어 있어야 합니다.

1. **Python 3.8+**: 개발 환경에 Python이 설치되어 있어야 합니다.
2. **Ollama**: Llama 3 모델 및 임베딩 모델을 로컬에서 실행하기 위해 Ollama가 설치되어 있어야 합니다.
    - [Ollama 다운로드 및 설치](https://www.google.com/url?sa=E&q=https%3A%2F%2Follama.com%2Fdownload)
    - **Llama 3 모델 다운로드**: ollama run llama3
    - **임베딩 모델 다운로드**: ollama pull nomic-embed-text
3. **네이버 개발자 센터 API 키**: 네이버 검색 API를 사용하기 위한 Client ID와 Client Secret을 발급받아야 합니다.
    - [네이버 개발자 센터](https://www.google.com/url?sa=E&q=https%3A%2F%2Fdevelopers.naver.com%2F)
    - 애플리케이션 등록 시 **"서비스 URL"**에는 http://localhost 또는 http://127.0.0.1을 입력합니다.
    - *"API 설정"**에서 **"검색" API**를 "사용함"으로 설정해야 합니다.
4. **Python 가상 환경 설정 (권장)**: code Bash
    
    downloadcontent_copy
    
    expand_less
    
    ```python
     python -m venv venv
    .\venv\Scripts\activate # Windows
    source venv/bin/activate # macOS/Linux
    ```
    
5. **Python 라이브러리 설치**: code Bash
    
    downloadcontent_copy
    
    expand_less
    
    ```python
    pip install dotenv langchain langchain-core langchain-community langchain-ollama chromadb
    ```
    

---

### 3. 시스템 아키텍처 (System Architecture)

이 RAG 시스템은 두 가지 주요 구성 요소로 나뉩니다.

1. **데이터 인제스트 파이프라인 (ingest_data.py)**:
    - 네이버 검색 API를 주기적으로 호출하여 최신 정보를 수집합니다.
    - 수집된 텍스트 데이터를 Document 객체로 변환합니다.
    - OllamaEmbeddings를 사용하여 Document를 벡터 임베딩으로 변환합니다.
    - 생성된 임베딩과 Document를 ChromaDB 벡터 데이터베이스에 저장합니다.
    - 이 스크립트를 주기적으로 실행하여 데이터베이스를 갱신합니다.
2. **질의 응답 파이프라인 (query_system.py)**:
    - ingest_data.py를 통해 구축된 ChromaDB 벡터 데이터베이스를 로드합니다.
    - 사용자의 질문이 들어오면 OllamaEmbeddings를 사용하여 질문을 임베딩합니다.
    - 벡터 데이터베이스에서 질문과 가장 관련성이 높은 문서를 검색합니다 (Retrieval).
    - 검색된 문서를 Llama 3 모델의 프롬프트 컨텍스트에 포함하여 답변을 생성합니다 (Generation).
