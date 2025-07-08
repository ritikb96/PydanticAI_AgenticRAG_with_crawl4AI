import os
import sys
import asyncio
import json
import requests
from dotenv import load_dotenv
from crawl4ai import BrowserConfig,CrawlerRunConfig,AsyncWebCrawler,CacheMode
from typing import List,Dict,Any
from xml.etree import ElementTree
from datetime import datetime,timezone
from urllib.parse import urlparse
from dataclasses import dataclass

from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)


@dataclass
class ProcessedChunk:
    url:str
    chunk_number:int
    title:str
    summary:str
    content:str
    metadata : Dict[str, Any]
    embedding : List[float]

def chunk_text(text:str,chunk_size: int= 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        #If we are the end of the text, we take all that's leftover
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        #try to find a code block boundary first ```
        chunk = text[start:end]
        codeblock = chunk.rfind('```')
        if codeblock != -1 and codeblock > chunk_size * .3:
            end = start + codeblock
        
        elif '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break

        elif '.' in chunk:
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.3:
                end = start + last_period

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = max(start+1,end)
    return chunks

async def get_title_and_summary(chunk:str,url:str) -> Dict[str,str]:
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

    try:
        response = await openai_client.chat.completions.create(
            model = os.getenv('LLM_MODEL','gpt-4o-mini'),
                messages = [
                {"role":"system","content":system_prompt},
                {"role":"user","content":f"URL:{url}\n\ncontent:\n{chunk[:1000]}...."}
    
            ],
            response_format = {"type":"json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return{"title":"Error processing title","processsing":"Error processsing summary"}

async def get_embedding(text:str) -> List[float]:
    """Getting embedding vector from openai"""
    try:
        response = await openai_client.embeddings.create(
            model = "text-embedding-3-small",
            input = text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error embedding the text: {e}")
        return [0] * 1536
    
async def process_chunk(chunk:str,chunk_number:int,url:str) -> ProcessedChunk:
    """Process a single chunk of data"""
    extracted = await get_title_and_summary(chunk,url)

    #get embedding
    embedding = await get_embedding(chunk)

    metadata = {
        "source":"PydanticAI_docs",
        "size":len(chunk),
        "crawled_at":datetime.now(timezone.utc).isoformat(),
        "url_path":urlparse(url).path
    }
    return ProcessedChunk(
        url = url,
        chunk_number = chunk_number,
        title = extracted['title'],
        summary = extracted['summary'],
        content = chunk,
        metadata = metadata,
        embedding = embedding
    )

async def insert_chunk(chunk:ProcessedChunk):
    """Insert chunk into the supabase database"""
    try:
        data = {
            "url":chunk.url,
            "chunk_number":chunk.chunk_number,
            "title":chunk.title,
            "summary":chunk.summary,
            "content":chunk.content,
            "metadata":chunk.metadata,
            "embedding":chunk.embedding
        }
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error getting inserting chunk:{e}")
        return None
    
async def process_and_store_document(url:str,markdown:str):
    """"process documents and store the documents in parallel"""
    chunks = chunk_text(markdown)

    #process the chunks
    tasks  = [
        process_chunk(chunk,i,url)
        for i,chunk in enumerate(chunks)

    ]
    processed_chunk = await asyncio.gather(*tasks)

    #insert chunks in parallel
    insert_chunks = [
        insert_chunk(chunk)
        for chunk in processed_chunk
    ]
    await asyncio.gather(*insert_chunks)

async def crawl_parallel(urls:List[str],max_concurrency:int=5):
    """crawl multiple urls in parallel with concurrency limits"""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=['--disable-gpu','--disable-dev-shm-usage','--no-sandbox']
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    #create crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        #create semaphore for concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_url(url:str):
            async with  semaphore:
                result = await crawler.arun(
                    config = crawl_config,
                    url = url,
                    session_id = "session1"
        )
            if result.success:
                print(f"successfully crawled the {url}")
                await process_and_store_document(url,result.markdown.raw_markdown)

            else:
                print(f"failed url:{url}:{result.error_message}")
        
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_pydantic_ai_docs_urls() -> List[str]:
    """"lets get all the urls from pydantic ai"""
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()

        #parse the xml
        root = ElementTree.fromstring(response.content)

        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]

        return urls
    except Exception as e:
        print(f"Error fetching urls from sitemap:{e}")
        return []
    
async def main():
    urls = get_pydantic_ai_docs_urls()
    if not urls:
        print("No urls are found to crawl")
        return
    print(f"found {len(urls)} URLs from the crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())






     

    

