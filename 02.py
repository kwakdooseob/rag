# .env 파일에서 환경 변수를 로드
from dotenv import load_dotenv
load_dotenv()

# 필요한 라이브러리 임포트
import os
from langchain_ollama import OllamaLLM
# create_tool_calling_agent 대신 initialize_agent와 AgentType 사용
from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain_naver_community.utils import NaverSearchAPIWrapper

# --- 1단계: 네이버 API 설정 ---
naver_client_id = os.environ.get("NAVER_CLIENT_ID")
naver_client_secret = os.environ.get("NAVER_CLIENT_SECRET")

if not naver_client_id or not naver_client_secret:
    print("네이버 API 인증 정보(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)를 환경 변수에 설정해주세요.")
    exit()

search_wrapper = NaverSearchAPIWrapper(
    naver_client_id=naver_client_id,
    naver_client_secret=naver_client_secret
)

# --- 2단계: Llama 3 모델 및 Agent 구성 ---
print("2단계: Llama 3 모델 및 Agent 구성...")
ollama_url = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11435')
llm = OllamaLLM(model="llama3", temperature=0, base_url=ollama_url)

# Agent의 도구(Tool) 정의
naver_search_tool = Tool(
    name="naver_search",
    func=lambda query: search_wrapper.results(query, search_type='blog'), # search_type='blog' 강제
    description="네이버 검색을 통해 최신 정보를 검색합니다. '뉴스', '블로그', '웹사이트', '이미지' 등 다양한 정보를 찾을 수 있습니다."
)
# Agent가 사용할 도구 목록
tools = [naver_search_tool]

# Agent 체인 구축
# initialize_agent 함수를 사용하여 ReAct 에이전트를 정의
# ReAct는 llm에 바인딩할 필요가 없습니다.
agent_executor = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True
)

print("Agent 체인 구축 완료. 이제 네이버 검색을 사용할 수 있습니다.")

# --- 3단계: 질의 응답 ---
print("\n3단계: 질의 응답...")
while True:
    question = input("질문하세요 (종료하려면 'exit' 입력): ")
    if question.lower() == 'exit':
        break
    
    try:
        result = agent_executor.invoke({"input": question})
        print("\n--- 질의 응답 결과 ---")
        print(f"질문: {question}")
        print(f"답변: {result['output']}")
        print("----------------------\n")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")