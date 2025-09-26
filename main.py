# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import os
import sys
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate # PromptTemplate 추가
from langchain.chains import create_retrieval_chain, LLMChain # LLMChain 추가
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Contextual Compression을 위한 추가 임포트 ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
# ----------------------------------------------------

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
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_HOST)
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
        
        # --- 기본 리트리버 설정 (더 많은 문서를 가져와 압축기에 전달) ---
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20}) # 일단 20개 가져오기
        
        # --- ContextualCompressionRetriever 설정 (Custom Prompt 적용) ---
        # 1. 압축기(Extractor)가 사용할 LLM 프롬프트 정의
        _compress_template = """다음 문서와 질문을 기반으로, 질문에 답변하는 데 가장 관련성이 높은 부분만 문서에서 추출하세요.
        문서가 질문과 관련된 정보를 포함하지 않는다면, "NO_OUTPUT"이라고 응답하세요.
        질문에 직접 답하지 마세요. 오직 문서에서 관련된 스니펫(짧은 구절)만 추출하세요.
        
        문서:
        {context}
        
        질문: {question}
        
        관련 스니펫:"""
        COMPRESSOR_PROMPT = PromptTemplate(template=_compress_template, input_variables=["context", "question"])

        # 2. 압축기 LLM을 위한 LLMChain 생성 (커스텀 프롬프트 사용)
        compressor_llm_chain = LLMChain(llm=llm, prompt=COMPRESSOR_PROMPT)

        # 3. LLMChainExtractor에 커스텀 LLMChain 연결
        compressor = LLMChainExtractor(llm_chain=compressor_llm_chain) # 커스텀 LLMChain 연결

        # 4. ContextualCompressionRetriever 생성
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )
        
        print(f"✅ Chroma VectorStore '{CHROMA_DB_PATH}' 로드 및 ContextualCompressionRetriever (k=20, Custom Compressor Prompt) 설정 완료.")
    except Exception as e:
        print(f"❌ Chroma VectorStore 로드 실패: {e}")
        sys.exit(1)

    # --- 3. RAG 프롬프트 정의 (LLM이 답변을 더 잘 생성하도록 유도) ---
    prompt = ChatPromptTemplate.from_template("""
    당신은 사용자에게 정보를 제공하는 유능한 어시스턴트입니다.
    주어진 문맥(context)을 주의 깊게 읽고, 질문에 가장 적합한 정보를 찾아 요약하여 답변하세요.
    **반드시 주어진 문맥에서 찾은 정보를 바탕으로 답변해야 하며, 당신의 사전 지식을 추가하여 답변하세요.**
    만약 주어진 문맥에 질문에 대한 답변 정보가 전혀 없다면,
    **"제공된 정보만으로는 답변하기 어렵습니다."** 라고 명확하게 말하세요.
    질문과 관련된 정보가 문맥에 있다면 그 정보를 바탕으로 답변합니다.
    **모든 답변은 꼭 한국어(korean) 으로 답변하세요.**

    [컨텍스트]
    {context}
    
    [질문]
    {input}
    
    [답변]
    """)

    # --- 4. RAG 체인 구성 ---
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(compression_retriever, document_chain) # 압축 리트리버 사용

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
            
            # 디버깅을 위해 검색된 문서도 함께 출력합니다.
            print("\n--- 검색된 문서 (Context) 미리보기 (LLM에 전달된 최종 컨텍스트) ---")
            if 'context' in response and response['context']:
                for i, doc in enumerate(response['context']):
                    print(f"  --- 검색 문서 {i+1} ---")
                    print(f"    제목: {doc.metadata.get('title', 'N/A')}")
                    print(f"    출처: {doc.metadata.get('source', 'N/A')}")
                    print(f"    내용:\n{doc.page_content}\n") 
            else:
                print("  검색된 문서가 없습니다.")
            print("----------------------\n")

        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            print("Ollama 서버가 실행 중인지, 모델이 올바르게 로드되었는지 확인하세요.")

if __name__ == "__main__":
    main()