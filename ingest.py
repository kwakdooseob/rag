# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import sys
import time
import shutil
import requests
from bs4 import BeautifulSoup
import json
import re

# Selenium 관련 임포트
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException, NoSuchElementException, StaleElementReferenceException

# Langchain Document Loaders for local files
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader
# Note: For Unstructured loaders (e.g., UnstructuredWordDocumentLoader, UnstructuredExcelLoader),
# you generally need the 'unstructured' library installed.
# For .pdf, 'pypdf'
# For .docx, 'python-docx'
# For .csv, 'pandas' (often implicitly used by CSVLoader or via unstructured)
# For .xlsx, 'openpyxl' (often implicitly used by UnstructuredExcelLoader or via unstructured)


# Load environment variables from .env file
load_dotenv()

# --- Environment Variables ---
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
OLLAMA_EMBEDDING_MODEL = os.environ.get('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
CHROMA_DB_PATH = os.environ.get('CHROMA_DB_PATH', './chroma_db')

NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")

LOCAL_DOCS_PATH = os.environ.get('LOCAL_DOCS_PATH', './local_documents')
NAVER_BLOG_QUERIES_FILE = os.environ.get("NAVER_BLOG_QUERIES_FILE", "./naver_blog_queries.txt") # .env에서 쿼리 파일 경로 로드

# ChromeDriver path configuration
CHROMEDRIVER_PATH = os.path.join(os.path.dirname(__file__), 'chromedriver.exe') 

print(f"Ollama Host: {OLLAMA_HOST}")
print(f"Embedding Model: {OLLAMA_EMBEDDING_MODEL}")
print(f"Chroma DB Path: {CHROMA_DB_PATH}")
print(f"Naver Client ID (first 5 chars): {NAVER_CLIENT_ID[:5] if NAVER_CLIENT_ID else 'Not Set'}")
print(f"ChromeDriver Path: {CHROMEDRIVER_PATH}")
print(f"Local Documents Path: {LOCAL_DOCS_PATH}")
print(f"Naver Blog Queries File: {NAVER_BLOG_QUERIES_FILE}")


# Validate Naver API credentials
if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
    print("❌ 네이버 API 인증 정보(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)를 환경 변수에 설정해주세요.")
    sys.exit(1)

# Naver Search API Configuration
NAVER_BLOG_SEARCH_API_URL = "https://openapi.naver.com/v1/search/blog.json"
NAVER_API_HEADERS = {
    "X-Naver-Client-Id": NAVER_CLIENT_ID,
    "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
}

# --- Selenium WebDriver 초기화/종료 함수 ---
def get_new_driver():
    # print("Initializing Chrome WebDriver for scraping...") # 너무 자주 출력될 수 있어 주석 처리
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument('disable-gpu')
    options.add_argument('no-sandbox')
    options.add_argument('disable-dev-shm-usage')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    try:
        service = Service(executable_path=CHROMEDRIVER_PATH)
        driver_instance = webdriver.Chrome(service=service, options=options)
        # print("Chrome WebDriver initialized.") # 너무 자주 출력될 수 있어 주석 처리
        return driver_instance
    except WebDriverException as e:
        print(f"❌ ChromeDriver 초기화 실패: {e}")
        print(f"💡 ChromeDriver ({CHROMEDRIVER_PATH})가 올바른 경로에 있고, Chrome 브라우저 버전과 일치하는지 확인하세요.")
        return None

def quit_driver_safely(driver_instance):
    if driver_instance:
        try:
            driver_instance.quit()
            # print("Chrome WebDriver closed.") # 너무 자주 출력될 수 있어 주석 처리
        except Exception as e:
            print(f"⚠️ WebDriver 종료 중 오류 발생: {e}")

def scrape_blog_content_selenium(url: str) -> str:
    if "n.news.naver.com" in url or "tistory.com" in url or "daum.net" in url or "post.naver.com" in url:
        print(f"   ℹ️ 네이버 뉴스/타 플랫폼/네이버 포스트 링크는 스크래핑하지 않습니다: {url}")
        return ""
    
    if not "blog.naver.com" in url:
        print(f"   ℹ️ 네이버 블로그 도메인이 아니므로 스크래핑하지 않습니다: {url}")
        return ""

    browser = None
    content_text = ""

    try:
        browser = get_new_driver()
        if browser is None:
            return ""

        print(f"   🌐 브라우저: {url} 로드 중...")
        browser.get(url)
        WebDriverWait(browser, 15).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        time.sleep(2)

        target_html_source = browser.page_source
        current_url = browser.current_url

        try:
            iframe_ids = ['mainFrame', 'screenFrame'] 
            iframe_element = None
            for iframe_id in iframe_ids:
                try:
                    iframe_element = WebDriverWait(browser, 3).until(
                        EC.presence_of_element_located((By.ID, iframe_id))
                    )
                    if iframe_element:
                        break
                except TimeoutException:
                    continue

            if iframe_element:
                iframe_src = iframe_element.get_attribute('src')
                if iframe_src.startswith('//'):
                    iframe_url = "https:" + iframe_src
                elif iframe_src.startswith('/'):
                    iframe_url = "https://blog.naver.com" + iframe_src
                else:
                    iframe_url = iframe_src

                print(f"   🔎 iframe (ID: {iframe_element.get_attribute('id')}) 발견. URL: {iframe_url} 전환 시도 중...")
                
                if iframe_url and "blog.naver.com" in iframe_url and iframe_url != current_url:
                    browser.switch_to.frame(iframe_element)
                    WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
                    time.sleep(1)
                    target_html_source = browser.page_source
                    print(f"   ✅ iframe 내부로 전환 성공.")
                else:
                    print(f"   ⚠️ iframe URL이 유효하지 않거나 메인 페이지와 동일. 메인 페이지 HTML 사용: {iframe_url}")
            else:
                print(f"   ⚠️ 페이지에서 적합한 iframe을 찾지 못했습니다. 메인 페이지 HTML 사용.")
        except StaleElementReferenceException:
            print(f"   ⚠️ StaleElementReferenceException 발생. 페이지 리로드 후 재시도.")
            browser.get(url)
            WebDriverWait(browser, 15).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            time.sleep(2)
            target_html_source = browser.page_source
        except Exception as e:
            print(f"   ⚠️ iframe 처리 중 오류 발생: {e}. 메인 페이지 HTML 사용.")
            target_html_source = browser.page_source

        soup_to_parse = BeautifulSoup(target_html_source, 'html.parser')

        content_selectors = [
            'div.se-main-container',
            'div.se-viewer',
            'div#postListBody',
            'div#post-view',
            'div.blog_post',
            'div.se_component_wrap',
            'div.post-content',
            'div.post_ct',
            'div.blog_content',
            'div.area_view',
            'div.section_article',
            'div[id^="post-area"]',
            'div.se_component'
        ]
        
        content_element_soup = None
        for selector in content_selectors:
            try:
                content_element_soup = soup_to_parse.select_one(selector)
                if content_element_soup:
                    print(f"   ✅ 본문 콘텐츠 영역 발견 (CSS: {selector})")
                    break
            except Exception:
                continue
        
        if content_element_soup:
            for unwanted_tag in content_element_soup(['script', 'style', 'noscript', 'img', 'a', 'iframe', 'video', 'audio', 'header', 'footer', 'nav', 'form']):
                unwanted_tag.extract()
            
            content_text = content_element_soup.get_text(separator='\n', strip=True)
            content_text = re.sub(r'\n\s*\n', '\n\n', content_text).strip()
            
        else:
            print(f"   ⚠️ 특정 본문 콘텐츠 영역을 찾지 못했습니다. 일반적인 텍스트 태그 추출 시도: {url}")
            
            possible_main_content_area = soup_to_parse.find('div', class_='se-main-area') or \
                                         soup_to_parse.find('div', id='viewTypeSelector') or \
                                         soup_to_parse.find('div', class_='post-content') or \
                                         soup_to_parse.find('div', class_='wrap_blogview') or \
                                         soup_to_parse.find('body')

            if possible_main_content_area:
                for unwanted_tag in possible_main_content_area(['script', 'style', 'noscript', 'img', 'a', 'iframe', 'video', 'audio', 'header', 'footer', 'nav', 'form']):
                    unwanted_tag.extract()
                
                paragraphs = possible_main_content_area.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li', 'span', 'strong'])
                if paragraphs:
                    content_text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                    content_text = re.sub(r'\n\s*\n', '\n\n', content_text).strip()
                    if content_text:
                        print(f"   ℹ️ 일반적인 텍스트 태그로 텍스트를 일부 추출했습니다.")
            
            if not content_text:
                print(f"   ❌ 본문 내용 스크래핑 최종 실패: {url}")

        return content_text.strip()
            
    except TimeoutException:
        print(f"   ⚠️ 페이지 로드 시간 초과 ({url})")
        return ""
    except WebDriverException as e:
        print(f"   ⚠️ WebDriver 오류 ({url}): {e}")
        return ""
    except Exception as e:
        print(f"   ⚠️ 블로그 내용 스크래핑 중 알 수 없는 오류 ({url}): {e}. 타입: {type(e).__name__}")
        return ""
    finally:
        try:
            if browser:
                browser.switch_to.default_content() 
        except Exception as e:
            print(f"   ⚠️ 드라이버 컨텍스트 초기화 중 오류 발생: {e}")
        quit_driver_safely(browser)


def retrieve_and_split_naver_blog_contents(queries: list, num_results_per_query: int = 5):
    if not queries:
        print("💡 검색할 쿼리가 제공되지 않았습니다.")
        return []

    print("\n--- 1. 네이버 블로그 데이터 검색 및 로드 및 본문 스크래핑 ---")
    all_documents = []
    
    for query in queries:
        query = query.strip()
        if not query: continue

        print(f"\n'{query}'에 대한 네이버 블로그 검색 중...")
        current_query_results = []
        start_index = 1 

        while len(current_query_results) < num_results_per_query:
            display_count = min(num_results_per_query - len(current_query_results), 100)
            if display_count <= 0:
                break

            params = {
                "query": query,
                "display": display_count,
                "start": start_index,
                "sort": "sim"
            }
            
            try:
                response = requests.get(NAVER_BLOG_SEARCH_API_URL, headers=NAVER_API_HEADERS, params=params, timeout=10)
                response.raise_for_status()
                search_data = json.loads(response.text)
                
                items = search_data.get('items', [])
                
                unique_items = []
                # 현재 검색 세션 내에서만 중복을 방지합니다.
                existing_links_in_session = {item.get('link') for item in current_query_results}
                
                for item in items:
                    clean_link = item.get('link')
                    if clean_link and not clean_link.startswith(('http://', 'https://')):
                        continue
                    if clean_link not in existing_links_in_session:
                        unique_items.append(item)
                        existing_links_in_session.add(clean_link)
                current_query_results.extend(unique_items)
                
                if not items or len(items) < display_count:
                    break
                
                start_index += display_count
                time.sleep(0.1)
            except requests.exceptions.RequestException as e:
                print(f"   ❌ 네이버 블로그 검색 API 요청 오류 ('{query}'): {e}")
                break
            except json.JSONDecodeError as e:
                print(f"   ❌ 네이버 블로그 검색 API 응답 JSON 파싱 오류 ('{query}'): {e}")
                break
            except Exception as e:
                print(f"   ❌ 네이버 블로그 검색 중 알 수 없는 오류 ('{query}'): {e}")
                break
        
        if not current_query_results:
            print(f"'{query}'에 대한 블로그 검색 결과가 없습니다.")
            continue

        for i, res in enumerate(current_query_results):
            blog_link = res.get("link")
            
            title_clean = res.get('title', '').replace("<b>", "").replace("</b>", "").replace("&quot;", "\"").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            desc_clean = res.get('description', '').replace("<b>", "").replace("</b>", "").replace("&quot;", "\"").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")

            if not blog_link or not blog_link.startswith(('http://', 'https://')):
                print(f"   ⚠️ 유효하지 않은 블로그 링크 건너뛰기: {blog_link}")
                content = f"제목: {title_clean}\n설명: {desc_clean}\n(유효하지 않은 링크)"
                doc = Document(page_content=content, metadata={"source": "invalid_link", "title": title_clean, "query": query, "pub_date": res.get("postdate", ""), "source_type": "blog"})
                all_documents.append(doc)
                continue 

            print(f"   🔗 '{title_clean}' ({blog_link}) 본문 스크래핑 시도 중...")
            full_blog_content = scrape_blog_content_selenium(blog_link) 
            
            content = f"제목: {title_clean}\n설명: {desc_clean}\n"
            if full_blog_content:
                content += f"본문:\n{full_blog_content}"
            else:
                content += "(본문 내용 스크래핑 실패 또는 없음)"

            doc = Document(
                page_content=content,
                metadata={
                    "source": blog_link,
                    "title": title_clean,
                    "query": query,
                    "pub_date": res.get("postdate", ""),
                    "source_type": "blog" # 메타데이터에 source_type 추가
                }
            )
            all_documents.append(doc)
            time.sleep(1)
        print(f"'{query}'에 대해 {len(current_query_results)}개의 블로그 검색 결과를 처리했습니다.")
        time.sleep(2)

    if not all_documents:
        print("⛔ 검색 및 스크래핑된 문서가 없어 인덱싱을 진행할 수 없습니다. 검색 쿼리 및 네이버 API 설정을 확인하세요.")
        return []

    return all_documents


def load_local_documents_and_split(local_docs_path: str):
    """
    Loads documents from a local directory.
    Supports .txt, .pdf, .docx, .csv.
    """
    print(f"\n--- 로컬 문서 '{local_docs_path}' 로드 및 분할 ---")
    
    if not os.path.exists(local_docs_path):
        print(f"❌ 로컬 문서 경로 '{local_docs_path}'를 찾을 수 없습니다. 폴더를 생성하거나 경로를 확인하세요.")
        return []

    documents = []
    
    loader_mapping = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader, 
        ".csv": CSVLoader,
        # ".xlsx": UnstructuredExcelLoader # Requires 'unstructured' and 'openpyxl'
    }
    
    for root, _, files in os.walk(local_docs_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_extension = os.path.splitext(file_name)[1].lower()

            loader_class = loader_mapping.get(file_extension)
            if loader_class:
                print(f"   📄 로컬 문서 로드 중: {file_name} (유형: {file_extension})")
                try:
                    loader = None
                    if file_extension == '.txt':
                        # Try multiple encodings for TXT files
                        encodings = ['utf-8', 'cp949', 'euc-kr']
                        for enc in encodings:
                            try:
                                temp_loader = TextLoader(file_path, encoding=enc)
                                _ = temp_loader.load() # Test load
                                loader = temp_loader
                                print(f"      ℹ️ '{file_name}' ({enc} 인코딩 성공)")
                                break # Found a working encoding
                            except UnicodeDecodeError:
                                print(f"      ℹ️ '{file_name}' ({enc} 인코딩 실패), 다음 시도...")
                            except Exception as e:
                                print(f"   ❌ '{file_name}' 텍스트 로드 중 알 수 없는 오류 ({enc}): {e}")
                                break # Stop trying encodings for this file on other errors
                        if loader is None: # If no suitable loader was found after all encoding attempts
                            print(f"   ❌ '{file_name}' 모든 인코딩 시도 실패. 파일 건너뛰기.")
                            continue # Skip this file and go to next file_name in loop
                    else: # For other document types, just use the the loader_class
                        loader = loader_class(file_path)

                    if loader: # Only proceed if a loader was successfully determined
                        loaded_docs = loader.load()
                        for doc in loaded_docs:
                            doc.metadata["source"] = file_path
                            doc.metadata["title"] = file_name
                            doc.metadata["source_type"] = "local_doc" # Add source_type metadata
                            documents.append(doc)
                        print(f"   ✅ 로컬 문서 로드 완료: {file_name}")
                except Exception as e:
                    print(f"   ❌ 로컬 문서 로드 중 오류 발생 ({file_name}): {e}")
            else:
                print(f"   ⚠️ 지원하지 않는 파일 형식 건너뛰기: {file_name}")
    
    if not documents:
        print(f"💡 로컬 문서 폴더 '{local_docs_path}'에 로드할 문서가 없습니다.")
        return []

    return documents # Return Document objects


def embed_and_store(chunks, db_path, embeddings_function):
    print("\n--- 2. 임베딩 모델 로드 및 벡터 데이터베이스 생성/업데이트 ---")
    try:
        vectorstore = None
        
        # Check if the DB directory exists and is not empty
        if os.path.exists(db_path) and os.listdir(db_path):
            try:
                # Try to load existing DB
                vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings_function)
                print(f"✅ 기존 Chroma DB '{db_path}' 로드 완료. 새 청크를 추가합니다.")
                
                # Basic duplicate prevention: check if source already exists in DB
                existing_sources = set()
                try:
                    # Fetch only metadata to avoid loading all document contents into memory, which can be slow for large DBs.
                    # We assume 'source' metadata is unique enough for deduplication.
                    existing_docs_in_db = vectorstore.get(include=['metadatas'])
                    for meta in existing_docs_in_db['metadatas']:
                        existing_sources.add(meta.get('source'))
                except Exception as e:
                    print(f"   ⚠️ 기존 DB 문서 소스 로드 중 오류 발생 (무시하고 진행): {e}")

                new_chunks = []
                for chunk in chunks:
                    if chunk.metadata.get('source') not in existing_sources:
                        new_chunks.append(chunk)
                    else:
                        print(f"   ℹ️ 중복된 문서 청크 건너뛰기: {chunk.metadata.get('source')} - {chunk.metadata.get('title')}")
                
                if new_chunks:
                    vectorstore.add_documents(new_chunks)
                    print(f"✅ {len(new_chunks)}개의 새 청크가 '{db_path}' 경로의 Chroma DB에 성공적으로 추가되었습니다.")
                else:
                    print("💡 새로 추가할 중복되지 않은 청크가 없습니다.")

            except Exception as e:
                print(f"   ⚠️ 기존 Chroma DB 로드 중 오류 발생 ({e}). 새로운 DB를 생성합니다.")
                # If loading existing DB fails, create a new one
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings_function,
                    persist_directory=db_path
                )
                print(f"✅ {len(chunks)}개의 청크로 새로운 Chroma DB '{db_path}'가 성공적으로 생성되었습니다.")
        else:
            print(f"💡 Chroma DB '{db_path}'가 존재하지 않거나 비어있습니다. 새로운 DB를 생성합니다.")
            # If DB does not exist or is empty, create a new one
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings_function,
                persist_directory=db_path
            )
            print(f"✅ {len(chunks)}개의 청크로 새로운 Chroma DB '{db_path}'가 성공적으로 생성되었습니다.")

    except Exception as e:
        print(f"❌ Chroma DB에 문서 저장 중 오류 발생: {e}")
        sys.exit(1)
    finally:
        # ChromaDB 0.4.x onwards automatically persists, so explicit .persist() is not strictly needed.
        pass


if __name__ == "__main__":
    print("--- RAG 데이터 인덱싱 스크립트 시작 ---")
    
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        print(f"Ollama Embedding Model '{OLLAMA_EMBEDDING_MODEL}' 로드 완료.")
    except Exception as e:
        print(f"❌ Ollama Embedding Model 로드 실패: {e}")
        print("💡 Ollama 서버가 실행 중인지, 지정된 모델이 설치되어 있는지 확인하세요.")
        print(f"   예: ollama serve 실행 후, ollama pull {OLLAMA_EMBEDDING_MODEL}")
        sys.exit(1)

    try:
        # .env에서 검색 쿼리 파일 경로를 가져와 리스트로 파싱
        def load_queries_from_file(file_path):
            queries = []
            if not os.path.exists(file_path):
                print(f"⚠️ 경고: 쿼리 파일 '{file_path}'를 찾을 수 없습니다.")
                return []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        query = line.strip()
                        if query and not query.startswith('#'): # 빈 줄 및 주석 제거
                            queries.append(query)
                print(f"✅ 쿼리 파일 '{file_path}'에서 {len(queries)}개의 키워드 로드 완료.")
            except Exception as e:
                print(f"❌ 쿼리 파일 '{file_path}' 로드 중 오류 발생: {e}")
            return queries
        
        search_queries_list = load_queries_from_file(NAVER_BLOG_QUERIES_FILE)

        if not search_queries_list:
            print("⚠️ NAVER_BLOG_QUERIES_FILE에 검색 키워드가 설정되지 않았습니다. 블로그 검색을 건너뜁니다.")
        
        while True:
            final_documents_to_process = [] # 청크로 분할할 최종 Document 객체 리스트
            
            indexing_choice = input(
                "\n어떤 소스의 데이터를 인덱싱할까요? (1: 블로그, 2: 로컬 문서, 3: 둘 다, exit: 종료): "
            ).strip().lower()

            if indexing_choice == 'exit':
                break
            elif indexing_choice == '1' or indexing_choice == '블로그':
                if not search_queries_list:
                    print("❌ 블로그 검색 키워드가 파일에 설정되지 않아 블로그 인덱싱을 진행할 수 없습니다.")
                    continue
                num_blog_results_per_query = int(input(f"각 키워드당 몇 개의 블로그 검색 결과를 가져올까요? (최대 100개, 기본값: 10): ") or "10")
                if num_blog_results_per_query > 100: num_blog_results_per_query = 100
                elif num_blog_results_per_query < 1: num_blog_results_per_query = 10
                
                print(f"\n--- 미리 지정된 블로그 키워드로 인덱싱을 시작합니다 ---")
                print(f"검색 키워드: {search_queries_list}")
                print(f"각 키워드당 가져올 결과 수: {num_blog_results_per_query}")

                blog_docs = retrieve_and_split_naver_blog_contents(search_queries_list, num_blog_results_per_query)
                final_documents_to_process.extend(blog_docs)
            elif indexing_choice == '2' or indexing_choice == '로컬 문서':
                local_docs = load_local_documents_and_split(LOCAL_DOCS_PATH)
                final_documents_to_process.extend(local_docs)
            elif indexing_choice == '3' or indexing_choice == '둘 다':
                if not search_queries_list:
                    print("❌ 블로그 검색 키워드가 파일에 설정되지 않아 블로그 인덱싱을 진행할 수 없습니다. 로컬 문서만 인덱싱합니다.")
                    local_docs = load_local_documents_and_split(LOCAL_DOCS_PATH)
                    final_documents_to_process.extend(local_docs)
                    if not final_documents_to_process:
                        print("⛔ 인덱싱할 문서가 없습니다. 소스 선택 및 설정을 확인하세요.")
                    continue
                    
                num_blog_results_per_query = int(input(f"각 키워드당 몇 개의 블로그 검색 결과를 가져올까요? (최대 100개, 기본값: 10): ") or "10")
                if num_blog_results_per_query > 100: num_blog_results_per_query = 100
                elif num_blog_results_per_query < 1: num_blog_results_per_query = 10

                print(f"\n--- 미리 지정된 블로그 키워드로 인덱싱을 시작합니다 ---")
                print(f"검색 키워드: {search_queries_list}")
                print(f"각 키워드당 가져올 결과 수: {num_blog_results_per_query}")

                blog_docs = retrieve_and_split_naver_blog_contents(search_queries_list, num_blog_results_per_query)
                local_docs = load_local_documents_and_split(LOCAL_DOCS_PATH)
                final_documents_to_process.extend(blog_docs)
                final_documents_to_process.extend(local_docs)
            else:
                print("❌ 유효하지 않은 선택입니다. 다시 입력해주세요.")
                continue
            
            if final_documents_to_process:
                print(f"\n--- 총 {len(final_documents_to_process)}개의 문서를 청크로 분할합니다. ---")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100,
                    length_function=len,
                    add_start_index=True,
                )
                all_chunks = text_splitter.split_documents(final_documents_to_process)
                print(f"총 {len(all_chunks)}개의 청크가 생성되었습니다.")
                embed_and_store(all_chunks, CHROMA_DB_PATH, embeddings)
            else:
                print("⛔ 인덱싱할 문서가 없습니다. 소스 선택 및 설정을 확인하세요.")
            
            # 인덱싱 완료 후 다음 선택지를 위해 final_documents_to_process 초기화
            final_documents_to_process = []

        print("\n--- RAG 데이터 인덱싱 스크립트 종료 ---")
                
    except Exception as e:
        print(f"\n🚨 인덱싱 스크립트 실행 중 치명적인 오류 발생: {e}")
    finally:
        pass