# -*- coding: utf-8 -*-
import streamlit as st
from dotenv import load_dotenv
import os
import sys

# RAG components
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_retrieval_chain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables from .env file
load_dotenv()

# --- Environment Variables ---
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
OLLAMA_LLM_MODEL = os.environ.get('OLLAMA_LLM_MODEL', 'llama3')
OLLAMA_EMBEDDING_MODEL = os.environ.get('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
CHROMA_DB_PATH = os.environ.get('CHROMA_DB_PATH', './chroma_db')

# --- RAG Chain Setup (Streamlit 캐싱을 사용하여 앱 시작 시 한 번만 실행) ---
@st.cache_resource
def setup_rag_chain(source_filter=None): # source_filter 인자 추가
    """
    RAG 체인을 설정하고 초기화합니다.
    Streamlit의 @st.cache_resource를 사용하여 앱이 다시 로드되어도 한 번만 실행되도록 캐싱합니다.
    """
    st.info(f"--- RAG 체인 설정 시작 (필터: {source_filter if source_filter else '모두'}) ---")
    
    # 1. LLM 및 Embedding Model 로드
    try:
        llm = OllamaLLM(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_HOST, temperature=0.1)
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        st.success(f"✅ Ollama LLM ('{OLLAMA_LLM_MODEL}') 및 Embedding Model ('{OLLAMA_EMBEDDING_MODEL}') 로드 완료.")
    except Exception as e:
        st.error(f"❌ Ollama LLM 또는 Embedding Model 로드 실패: {e}")
        st.warning("💡 Ollama 서버가 실행 중인지, 지정된 모델들이 설치되어 있는지 확인하세요.")
        st.stop()

    # 2. Chroma VectorStore 로드
    if not os.path.exists(CHROMA_DB_PATH):
        st.error(f"❌ Chroma DB 경로 '{CHROMA_DB_PATH}'를 찾을 수 없습니다.")
        st.warning("💡 'python ingest.py'를 먼저 실행하여 벡터 데이터베이스를 생성해주세요.")
        st.stop()

    try:
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        
        # --- 리트리버 필터링 적용 ---
        search_kwargs = {"k": 5}
        if source_filter:
            search_kwargs["filter"] = {"source_type": source_filter}
            st.info(f"ℹ️ 리트리버 필터링 적용: source_type = '{source_filter}'")
            
        base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs) 
        
        # --- ContextualCompressionRetriever 설정 (Custom Prompt 적용) ---
        _compress_template = """다음 문서와 질문을 기반으로, 질문에 답변하는 데 가장 관련성이 높은 부분만 문서에서 추출하세요.
        문서가 질문과 관련된 정보를 포함하지 않는다면, "NO_OUTPUT"이라고 응답하세요.
        질문에 직접 답하지 마세요. 오직 문서에서 관련된 스니펫(짧은 구절)만 추출하세요.
        
        문서:
        {context}
        
        질문: {question}
        
        관련 스니펫:"""
        COMPRESSOR_PROMPT = PromptTemplate(template=_compress_template, input_variables=["context", "question"])

        compressor_llm_chain = LLMChain(llm=llm, prompt=COMPRESSOR_PROMPT)
        compressor = LLMChainExtractor(llm_chain=compressor_llm_chain) 
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )
        
        st.success(f"✅ Chroma VectorStore '{CHROMA_DB_PATH}' 로드 및 ContextualCompressionRetriever (k=20, Custom Compressor Prompt) 설정 완료.")
    except Exception as e:
        st.error(f"❌ Chroma VectorStore 로드 실패: {e}")
        st.stop()

    # 3. RAG 프롬프트 정의 (LLM이 답변을 더 잘 생성하도록 유도)
    rag_prompt = ChatPromptTemplate.from_template("""
    당신은 사용자에게 정보를 제공하는 유능한 어시스턴트입니다.
    주어진 문맥(context)을 주의 깊게 읽고, 질문에 가장 적합한 정보를 찾아 요약하여 답변하세요.
    **반드시 주어진 문맥에서 찾은 정보를 바탕으로 답변해야 하며, 당신의 사전 지식을 추가하지 마세요.**
    만약 주어진 문맥에 질문에 대한 답변 정보가 전혀 없다면,
    **"제공된 정보만으로는 답변하기 어렵습니다."** 라고 명확하게 말하세요.
    질문과 관련된 정보가 문맥에 있다면 그 정보를 바탕으로 답변합니다.

    [컨텍스트]
    {context}
    
    [질문]
    {input}
    
    [답변]
    """)

    # 4. RAG 체인 구성
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain = create_retrieval_chain(compression_retriever, document_chain)
    
    st.info("--- RAG 체인 구축 완료. 이제 질문하세요. ---")
    st.info("💡 이 시스템은 미리 인덱싱된 데이터를 기반으로 답변을 생성합니다.")
    return rag_chain

# --- Streamlit App UI ---
st.set_page_config(page_title="RAG 챗봇 (다중 소스)", page_icon="🤖", layout="wide")
st.title("🤖 RAG 챗봇 (다중 소스 기반)")
st.caption("네이버 블로그 및 로컬 문서 데이터를 기반으로 답변을 생성합니다.")

# Sidebar for source selection
with st.sidebar:
    st.header("🔍 검색 소스 선택")
    selected_source = st.radio(
        "어떤 소스에서 답변을 찾을까요?",
        options=["모두", "블로그", "로컬 문서"],
        key="source_selection"
    )
    st.markdown("---")
    st.info("새로운 소스 선택 시, RAG 체인이 재설정됩니다.")

# Map selection to filter value
source_filter_value = None
if selected_source == "블로그":
    source_filter_value = "blog"
elif selected_source == "로컬 문서":
    source_filter_value = "local_doc"

# Setup RAG chain based on selected source (runs once for each unique source_filter_value due to @st.cache_resource)
rag_chain = setup_rag_chain(source_filter_value)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"})

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input from user
if prompt := st.chat_input("질문하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                response = rag_chain.invoke({"input": prompt})
                ai_response = response['answer']
                st.markdown(ai_response)

                if 'context' in response and response['context']:
                    with st.expander("검색된 문서 (Context) 미리보기 (LLM에 전달된 최종 컨텍스트)"):
                        for i, doc in enumerate(response['context']):
                            st.write(f"**--- 검색 문서 {i+1} ({doc.metadata.get('source_type', 'N/A')}) ---**") # source_type 표시
                            st.write(f"**제목:** {doc.metadata.get('title', 'N/A')}")
                            st.write(f"**출처:** {doc.metadata.get('source', 'N/A')}")
                            st.write(f"**내용:** {doc.page_content}")
                            st.markdown("---")
                else:
                    st.info("검색된 문서가 없습니다.")

            except Exception as e:
                ai_response = f"죄송합니다. 답변을 생성하는 중 오류가 발생했습니다: {e}"
                st.error(ai_response)
                st.warning("💡 Ollama 서버가 실행 중인지, 모델이 올바르게 로드되었는지 확인하세요.")
        
        st.session_state.messages.append({"role": "assistant", "content": ai_response})