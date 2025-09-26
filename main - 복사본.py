# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import os
import sys
# import from langchain_ollama package instead of langchain_community
from langchain_ollama import OllamaLLM # 변경됨
from langchain_ollama import OllamaEmbeddings # 변경됨

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# .env 파일에서 환경 변수를 로드
load_dotenv()

# --- 환경 변수 로드 ---
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
OLLAMA_LLM_MODEL = os.environ.get('OLLAMA_LLM_MODEL', 'llama3')
OLLAMA_EMBEDDING_MODEL = os.environ.get('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
CHROMA_DB_PATH = os.environ.get('CHROMA_DB_PATH', './chroma_db')

print(f"Ollama Host: {OLLAMA_HOST}")
print(f"LLM Model: {OLLAMA_LLM_MODEL}")
print(f"Embedding Model: {OLLAMA_EMBEDDING_MODEL}")
print(f"Chroma DB Path: {CHROMA_DB_PATH}")

def main():
    print("\n--- RAG 기반 질의 응답 시스템 시작 ---")

    # --- 1. LLM 및 임베딩 모델 로드 ---
    try:
        llm = OllamaLLM(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_HOST) # 변경됨
        print(f"✅ Ollama LLM ('{OLLAMA_LLM_MODEL}') 및 Embedding Model ('{OLLAMA_EMBEDDING_MODEL}') 로드 완료.")
    except Exception as e:
        print(f"❌ Ollama LLM 또는 Embedding Model 로드 실패: {e}")
        print("💡 Ollama 서버가 실행 중인지, 지정된 모델들이 설치되어 있는지 확인하세요.")
        print(f"   예: ollama serve 실행 후, ollama pull {OLLAMA_LLM_MODEL} 및 ollama pull {OLLAMA_EMBEDDING_MODEL}")
        sys.exit(1)

    # --- 2. Chroma VectorStore 로드 ---
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"❌ Chroma DB 경로 '{CHROMA_DB_PATH}'를 찾을 수 없습니다.")
        print("💡 'python ingest.py'를 먼저 실행하여 벡터 데이터베이스를 생성해주세요.")
        sys.exit(1)

    try:
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        retriever = vectorstore.as_retriever()
        print(f"✅ Chroma VectorStore '{CHROMA_DB_PATH}' 로드 및 Retriever 설정 완료.")
    except Exception as e:
        print(f"❌ Chroma VectorStore 로드 실패: {e}")
        sys.exit(1)

    # --- 3. RAG 프롬프트 정의 ---
    prompt = ChatPromptTemplate.from_template("""
    주어진 문맥(context)을 사용하여 질문에 답변하세요.
    당신의 사전 지식은 사용하지 않고, 오직 주어진 문맥에서만 정보를 찾으세요.
    만약 문맥에서 답변을 찾을 수 없다면, "제공된 정보만으로는 답변하기 어렵습니다."라고 말하세요.

    <context>
    {context}
    </context>

    질문: {input}
    답변:
    """)

    # --- 4. RAG 체인 구성 ---
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    print("\n--- RAG 체인 구축 완료. 이제 질문하세요. ---")
    print("💡 이 시스템은 미리 인덱싱된 네이버 블로그 검색 결과에서 정보를 검색합니다.")

    # --- 5. 질의 응답 루프 ---
    while True:
        question = input("\n질문하세요 (종료하려면 'exit' 입력): ")
        if question.lower() == 'exit':
            break
        
        try:
            print(f"\n[Thinking...] 질문: {question}")
            response = rag_chain.invoke({"input": question})
            
            print("\n--- 질의 응답 결과 ---")
            print(f"답변: {response['answer']}")
            
            # 디버깅을 위해 검색된 문서도 함께 출력할 수 있습니다.
            # print("\n--- 검색된 문서 (Context) ---")
            # for doc in response['context']:
            #     print(f"Source: {doc.metadata.get('source', 'N/A')}")
            #     print(f"Title: {doc.metadata.get('title', 'N/A')}")
            #     print(f"Content: {doc.page_content[:150]}...\n")
            print("----------------------\n")

        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            print("Ollama 서버가 실행 중인지, 모델이 올바르게 로드되었는지 확인하세요.")

if __name__ == "__main__":
    main()