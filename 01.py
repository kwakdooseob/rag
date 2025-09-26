# 필요한 라이브러리 임포트
# 변경된 import 문
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from pathlib import Path
import os

# --- 1단계: 데이터 불러오기 및 전처리 ---
print("1단계: 데이터 불러오기 및 전처리...")
file_path = "data.txt"

if not Path(file_path).exists():
    raise FileNotFoundError(f"Error: {file_path} 파일이 존재하지 않습니다. 먼저 파일을 만들어주세요.")

loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

print(f"원본 문서 개수: {len(documents)}")
print(f"분할된 청크 개수: {len(docs)}")

# --- 2단계: 임베딩 및 벡터 데이터베이스 생성 또는 불러오기 ---
print("\n2단계: 임베딩 및 벡터 데이터베이스 생성...")
db_path = "faiss_index"

if os.path.exists(db_path):
    print("기존 벡터 데이터베이스를 로드합니다.")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
else:
    print("새로운 벡터 데이터베이스를 생성합니다.")
    model_name = "BAAI/bge-m3"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(db_path)
    print("FAISS 벡터 데이터베이스 생성 및 저장 완료.")


# --- 3단계: Llama 3 모델 연결 및 RAG 체인 구축 ---
print("\n3단계: Llama 3 모델 연결 및 RAG 체인 구축...")
ollama_url = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11435')

#llm = Ollama(model="llama3", temperature=0, base_url=ollama_url)
llm = Ollama(model="llama3:8b-instruct-q4_K_M", temperature=0, base_url=ollama_url)

# RAG 체인 구축
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)
print("RAG 체인 구축 완료.")

# --- 4단계: 질의 응답 ---
print("\n4단계: 질의 응답...")
while True:
    question = input("질문하세요 (종료하려면 'exit' 입력): ")
    if question.lower() == 'exit':
        break
    
    result = qa_chain.invoke({"query": question})

    print("\n--- 질의 응답 결과 ---")
    print(f"질문: {question}")
    print(f"답변: {result['result']}")
    
    if 'source_documents' in result and result['source_documents']:
        print("\n--- 참고 문서 ---")
        for doc in result['source_documents']:
            print(f"- 출처 문서: {doc.page_content}")
            print(f"- 문서 메타데이터: {doc.metadata}")
    print("----------------------\n")