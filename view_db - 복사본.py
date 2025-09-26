# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import sys

# .env 파일에서 환경 변수를 로드
load_dotenv()

# --- 환경 변수 로드 ---
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
OLLAMA_EMBEDDING_MODEL = os.environ.get('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
CHROMA_DB_PATH = os.environ.get('CHROMA_DB_PATH', './chroma_db')

print(f"Ollama Host: {OLLAMA_HOST}")
print(f"Embedding Model: {OLLAMA_EMBEDDING_MODEL}")
print(f"Chroma DB Path: {CHROMA_DB_PATH}")

def view_chroma_db():
    print("\n--- Chroma DB 데이터 조회 스크립트 시작 ---")

    # 1. 임베딩 모델 로드 (DB 로드 시 필요)
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        print(f"✅ Ollama Embedding Model '{OLLAMA_EMBEDDING_MODEL}' 로드 완료.")
    except Exception as e:
        print(f"❌ Ollama Embedding Model 로드 실패: {e}")
        print("💡 Ollama 서버가 실행 중인지, 지정된 모델이 설치되어 있는지 확인하세요.")
        sys.exit(1)

    # 2. Chroma DB 경로 확인
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"❌ Chroma DB 경로 '{CHROMA_DB_PATH}'를 찾을 수 없습니다.")
        print("💡 'python ingest.py'를 먼저 실행하여 벡터 데이터베이스를 생성해주세요.")
        sys.exit(1)

    # 3. Chroma DB 로드
    try:
        # persist_directory를 지정하여 기존 DB를 로드합니다.
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        print(f"✅ Chroma DB '{CHROMA_DB_PATH}' 로드 완료.")
        
        # 컬렉션 정보 가져오기 (Chroma DB가 내부적으로 사용하는 컬렉션)
        # ChromaDB는 기본적으로 'langchain'이라는 컬렉션에 문서를 저장합니다.
        # 실제 컬렉션 이름을 확인하려면, `vectorstore._collection.get()`을 사용해야 합니다.
        # 하지만, `Chroma` 클래스가 래핑하고 있으므로, `.get()` 메서드를 통해 직접 접근합니다.
        
        # 모든 문서의 ID와 메타데이터, 그리고 (선택적으로) 콘텐츠를 가져옵니다.
        # limit 파라미터를 사용하여 가져올 문서의 수를 제한할 수 있습니다.
        # include=['documents', 'metadatas']는 문서 내용과 메타데이터를 포함하라는 의미입니다.
        db_contents = vectorstore.get(
            ids=None, # 모든 ID (None이면 모두 가져옴)
            include=['documents', 'metadatas'] 
            # 'embeddings'를 포함하면 임베딩 벡터까지 가져올 수 있지만, 출력이 매우 길어질 수 있습니다.
        )

        print(f"\n--- Chroma DB에 저장된 총 {len(db_contents['ids'])}개의 문서 ---")

        if not db_contents['ids']:
            print("Chroma DB에 문서가 저장되어 있지 않습니다.")
            print("💡 'python ingest.py'를 실행하여 데이터를 인덱싱해주세요.")
            return

        # 저장된 문서들을 출력합니다.
        # 모든 문서를 출력하면 너무 길어질 수 있으므로, 처음 몇 개만 출력하도록 하겠습니다.
        num_to_display = min(len(db_contents['ids']), 5) # 최대 5개 문서만 출력

        for i in range(num_to_display):
            doc_id = db_contents['ids'][i]
            metadata = db_contents['metadatas'][i]
            document_content = db_contents['documents'][i]

            print(f"\n--- 문서 {i+1} (ID: {doc_id}) ---")
            print(f"  제목: {metadata.get('title', 'N/A')}")
            print(f"  출처(URL): {metadata.get('source', 'N/A')}")
            print(f"  검색 쿼리: {metadata.get('query', 'N/A')}")
            print(f"  작성일: {metadata.get('pub_date', 'N/A')}")
            print(f"  내용 (일부):\n    {document_content[:500]}...") # 내용이 길면 잘라서 출력
            print("---------------------------------------")
        
        if len(db_contents['ids']) > num_to_display:
            print(f"\n... (총 {len(db_contents['ids'])}개 문서 중 {num_to_display}개만 출력되었습니다)")

    except Exception as e:
        print(f"❌ Chroma DB 로드 또는 조회 중 오류 발생: {e}")
        print("💡 Ollama 서버가 실행 중인지, Chroma DB 파일이 손상되지 않았는지 확인하세요.")
        sys.exit(1)

    print("\n--- Chroma DB 데이터 조회 스크립트 종료 ---")

if __name__ == "__main__":
    view_chroma_db()