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

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException, NoSuchElementException, StaleElementReferenceException

load_dotenv()

# --- Environment Variables ---
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
OLLAMA_EMBEDDING_MODEL = os.environ.get('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
CHROMA_DB_PATH = os.environ.get('CHROMA_DB_PATH', './chroma_db')

NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")

CHROMEDRIVER_PATH = os.path.join(os.path.dirname(__file__), 'chromedriver.exe') 

print(f"Ollama Host: {OLLAMA_HOST}")
print(f"Embedding Model: {OLLAMA_EMBEDDING_MODEL}")
print(f"Chroma DB Path: {CHROMA_DB_PATH}")
print(f"Naver Client ID (first 5 chars): {NAVER_CLIENT_ID[:5] if NAVER_CLIENT_ID else 'Not Set'}")
print(f"ChromeDriver Path: {CHROMEDRIVER_PATH}")

if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
    print("❌ 네이버 API 인증 정보(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)를 환경 변수에 설정해주세요.")
    sys.exit(1)

NAVER_BLOG_SEARCH_API_URL = "https://openapi.naver.com/v1/search/blog.json"
NAVER_API_HEADERS = {
    "X-Naver-Client-Id": NAVER_CLIENT_ID,
    "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
}

# --- Selenium WebDriver 초기화/종료 함수 (각 스크래핑마다 호출) ---
def get_new_driver():
    """새로운 Selenium WebDriver 인스턴스를 생성합니다."""
    print("Initializing Chrome WebDriver for scraping...")
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
        print("Chrome WebDriver initialized.")
        return driver_instance
    except WebDriverException as e:
        print(f"❌ ChromeDriver 초기화 실패: {e}")
        print(f"💡 ChromeDriver ({CHROMEDRIVER_PATH})가 올바른 경로에 있고, Chrome 브라우저 버전과 일치하는지 확인하세요.")
        return None

def quit_driver_safely(driver_instance):
    """WebDriver 인스턴스를 안전하게 닫습니다."""
    if driver_instance:
        try:
            driver_instance.quit()
            print("Chrome WebDriver closed.")
        except Exception as e:
            print(f"⚠️ WebDriver 종료 중 오류 발생: {e}")


# --- 웹 스크래핑 헬퍼 함수 (Selenium 기반) ---
def scrape_blog_content_selenium(url: str) -> str:
    """
    주어진 URL에서 네이버 블로그 본문 내용을 Selenium을 사용하여 스크래핑합니다.
    (동적 로딩 및 iframe 처리)
    """
    if "n.news.naver.com" in url or "tistory.com" in url or "daum.net" in url or "post.naver.com" in url:
        print(f"   ℹ️ 네이버 뉴스/타 플랫폼/네이버 포스트 링크는 스크래핑하지 않습니다: {url}")
        return ""
    
    if not "blog.naver.com" in url:
        print(f"   ℹ️ 네이버 블로그 도메인이 아니므로 스크래핑하지 않습니다: {url}")
        return ""

    browser = None
    content_text = ""

    try:
        browser = get_new_driver() # 각 스크래핑마다 새로운 드라이버 인스턴스 생성
        if browser is None:
            return "" # 드라이버 초기화 실패 시 빈 문자열 반환

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
        # iframe에서 전환했으면 원래 컨텍스트로 돌아가야 함 (현재는 driver를 바로 닫으므로 불필요)
        # try:
        #     browser.switch_to.default_content() 
        # except Exception as e:
        #     print(f"   ⚠️ 드라이버 컨텍스트 초기화 중 오류: {e}")
        quit_driver_safely(browser) # 각 스크래핑 후에 드라이버 닫기


def retrieve_and_split_naver_blog_contents(queries: list, num_results_per_query: int = 5):
    """
    Retrieves blog posts from Naver Search API, then scrapes their content using Selenium,
    and returns them as Document objects.
    """
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
                existing_links = {doc.metadata['source'] for doc in all_documents}
                for item in items:
                    clean_link = item.get('link')
                    if clean_link and not clean_link.startswith(('http://', 'https://')):
                        continue
                    if clean_link not in existing_links:
                        unique_items.append(item)
                        existing_links.add(clean_link)
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
                doc = Document(page_content=content, metadata={"source": "invalid_link", "title": title_clean, "query": query, "pub_date": res.get("postdate", "")})
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
                    "pub_date": res.get("postdate", "")
                }
            )
            all_documents.append(doc)
            time.sleep(1)
        print(f"'{query}'에 대해 {len(current_query_results)}개의 블로그 검색 결과를 처리했습니다.")
        time.sleep(2)

    if not all_documents:
        print("⛔ 검색 및 스크래핑된 문서가 없어 인덱싱을 진행할 수 없습니다. 검색 쿼리 및 네이버 API 설정을 확인하세요.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(all_documents)
    print(f"총 {len(all_documents)}개의 검색 및 스크래핑된 문서에서 {len(chunks)}개의 청크로 분할했습니다.")
    return chunks

def embed_and_store(chunks, db_path, embeddings_function):
    """
    Embeds fragmented text chunks and stores them in Chroma DB.
    """
    print("\n--- 2. 임베딩 모델 로드 및 벡터 데이터베이스 생성/업데이트 ---")
    try:
        if os.path.exists(db_path):
            print(f"기존 Chroma DB '{db_path}'를 삭제합니다.")
            shutil.rmtree(db_path)
            
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_function,
            persist_directory=db_path
        )
        print(f"✅ {len(chunks)}개의 청크가 '{db_path}' 경로의 Chroma DB에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"❌ Chroma DB에 문서 저장 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("--- RAG 데이터 인덱싱 스크립트 시작 ---")
    
    # 드라이버는 각 스크래핑마다 생성/종료하므로, 여기서 전역 드라이버를 미리 초기화할 필요 없음

    try:
        # Load embedding model
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        print(f"Ollama Embedding Model '{OLLAMA_EMBEDDING_MODEL}' 로드 완료.")
    except Exception as e:
        print(f"❌ Ollama Embedding Model 로드 실패: {e}")
        print("💡 Ollama 서버가 실행 중인지, 지정된 모델이 설치되어 있는지 확인하세요.")
        print(f"   예: ollama serve 실행 후, ollama pull {OLLAMA_EMBEDDING_MODEL}")
        # close_driver() # 이제 전역 드라이버가 없으므로 필요 없음
        sys.exit(1)

    try:
        while True:
            user_input_queries = input(
                "\n네이버 블로그에서 검색할 키워드를 쉼표(,)로 구분하여 입력하세요 (예: 맛집 추천, 제주도 여행, 파이썬): "
                "\n또는 'exit'를 입력하여 종료하고, 'clear'를 입력하여 기존 DB를 삭제하고 새로 시작할 수 있습니다: "
            )
            
            if user_input_queries.lower() == 'exit':
                break
            elif user_input_queries.lower() == 'clear':
                if os.path.exists(CHROMA_DB_PATH):
                    print(f"기존 Chroma DB '{CHROMA_DB_PATH}'를 삭제합니다.")
                    shutil.rmtree(CHROMA_DB_PATH)
                    print("Chroma DB가 초기화되었습니다. 새로운 검색어를 입력해주세요.")
                else:
                    print("삭제할 Chroma DB가 없습니다.")
                continue
            
            search_queries = [q.strip() for q in user_input_queries.split(',') if q.strip()]
            
            if not search_queries:
                print("❌ 유효한 검색어가 입력되지 않았습니다. 다시 시도해주세요.")
                continue

            num_blog_results_per_query = int(input(f"각 키워드당 몇 개의 블로그 검색 결과를 가져올까요? (최대 100개, 기본값: 5): ") or "5")
            if num_blog_results_per_query > 100:
                num_blog_results_per_query = 100
                print("💡 네이버 블로그 API는 한 키워드당 최대 100개의 검색 결과를 제공합니다. 100개로 제한합니다.")
            elif num_blog_results_per_query < 1:
                num_blog_results_per_query = 5
                print("💡 검색 결과 수는 최소 1개 이상이어야 합니다. 5개로 설정합니다.")


            all_chunks = retrieve_and_split_naver_blog_contents(search_queries, num_blog_results_per_query) 
            
            if all_chunks:
                embed_and_store(all_chunks, CHROMA_DB_PATH, embeddings)
            else:
                print("⛔ 인덱싱할 문서가 없습니다. 검색 쿼리 및 네이버 API 설정을 확인하세요.")
                
    except Exception as e:
        print(f"\n🚨 인덱싱 스크립트 실행 중 치명적인 오류 발생: {e}")
    finally:
        # close_driver() # 이제 각 스크래핑마다 드라이버를 닫으므로 이 부분은 필요 없음
        print("--- RAG 데이터 인덱싱 스크립트 종료 ---")