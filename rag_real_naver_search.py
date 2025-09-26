# .env 파일에서 환경 변수를 로드
from dotenv import load_dotenv
load_dotenv()

# 필요한 라이브러리 임포트
import os
from langchain_ollama import OllamaLLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_naver_community.utils import NaverSearchAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage # agent_scratchpad를 위해 필요

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
    func=search_wrapper.results, # 검색을 수행하는 메서드는 results입니다.
    description="""네이버 검색을 통해 최신 정보를 검색합니다.
    '뉴스', '블로그', '웹사이트', '이미지' 등 다양한 정보를 찾을 수 있습니다.
    사용자의 질문에 답변하기 위해 필요한 정보를 검색하고,
    검색 결과가 나오면 이를 바탕으로 최종 답변을 구성해야 합니다."""
)

# Agent가 사용할 도구 목록
tools = [naver_search_tool]

# 사용자 정의 프롬프트 생성 (RAG 지향)
prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 유능한 질의응답 어시스턴트입니다. 사용자의 질문에 정확하고 간결하게 답변하는 것이 목표입니다.
    다음은 당신이 사용할 수 있는 도구들입니다:
    {tools}
    이 도구들 중 하나를 사용해야 합니다: {tool_names}

    다음 규칙을 엄격히 따르세요:

    1.  **도구 사용:** `naver_search` 도구를 사용하여 사용자의 질문과 관련된 최신 정보를 검색합니다.
        (예시: Action: tool_name\nAction Input: query)
    2.  **정보 활용:** 검색 결과(`Observation`)가 주어지면, 해당 정보(특히 `title`과 `description` 필드)를 **종합하여** 사용자의 질문에 대한 답변을 구성하세요. 마치 주어진 검색 결과가 당신의 지식 소스인 것처럼 활용하세요.
    3.  **팩트 기반:** 답변은 **오직** 검색 결과에서 얻은 정보에만 기반해야 합니다. 당신의 사전 지식을 추가하지 마세요.
    4.  **정확성 및 한계:** 검색 결과에 명확한 답변이 없으면, "정보를 찾을 수 없었습니다" 또는 "제공된 정보만으로는 답변하기 어렵습니다"와 같이 명확하게 답변의 한계를 밝히세요.
    5.  **불필요한 반복 피하기:** 한 번 검색해서 충분한 정보를 얻었다면, 추가 검색을 시도하지 않고 바로 답변을 생성하세요.
    6.  **최종 답변 형식:** 모든 추론과 도구 사용이 끝난 후, 최종 답변은 반드시 'Final Answer: [여기에 최종 답변]' 형식으로 시작해야 합니다."""
    ),
    ("human", "{input}\n{agent_scratchpad}"), # MessagesPlaceholder 대신 여기에 포함
])

# create_react_agent로 에이전트 생성
agent = create_react_agent(llm, tools, prompt)

# AgentExecutor 생성 (파싱 오류 핸들링 추가)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

print("Agent 체인 구축 완료. 이제 네이버 검색을 사용할 수 있습니다.")

# --- 3단계: 질의 응답 ---
print("\n3단계: 질의 응답...")
while True:
    question = input("질문하세요 (종료하려면 'exit' 입력): ")
    if question.lower() == 'exit':
        break
    
    try:
        result = agent_executor.invoke({
            "input": question,
            "agent_scratchpad": [] # 이 부분을 추가/수정
        })
        print("\n--- 질의 응답 결과 ---")
        print(f"질문: {question}")
        print(f"답변: {result['output']}")
        print("----------------------\n")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")