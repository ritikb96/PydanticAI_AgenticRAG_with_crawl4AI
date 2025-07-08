from __future__ import annotations as __annotations
import os
import sys
import logfire
import httpx
import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv
from supabase import Client
from pydantic_ai import Agent,RunContext,ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from typing import List

load_dotenv()
llm = os.getenv('LLM_MODEL','gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    open_ai_client: AsyncOpenAI

system_prompt =  """
You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation to,
including examples, an API reference, and other resources to help you build Pydantic AI agents.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""

pydantic_ai_expert = Agent(
    model = model,
    deps_type= PydanticAIDeps,
    system_prompt= system_prompt,
    retries=2
)

async def get_embedding(text:str,openai_client:AsyncOpenAI) -> List[float]:
    try:
        response = await openai_client.embeddings.create(
            model='text-embedding-3-small',
            input = text

        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embeddings for the user input: {e}")
        return [0] * 1536
    
@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx:RunContext[PydanticAIDeps],user_query:str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        #lets first get embeddings
        query_embedding = await get_embedding(user_query,ctx.deps.open_ai_client)

        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding':query_embedding,
                'match_count':5,
                'filter': {'source':'PydanticAI_docs'}
            }
        ).execute()

        if not result.data:
            print(f"No relevant documents found")
        else:
            formatted_chunks = []
            for doc in result.data:
                chunk_text = f"""
                #{doc['title']}
                {doc['content']}
                    """
            formatted_chunks.append(chunk_text)
            return "\n\n--\n\n".join(formatted_chunks)
    except Exception as e:
        print(f"Error retrieving relevant documents from the knowledge base: {e}")
        return "Error retrieving relevant documents: str{e}"

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx:RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        result = ctx.deps.supabase.from_('site_pages')\
        .select('url')\
        .eq('metadata->>source','PydanticAI_docs')\
        .execute()

        if not result.data:
            return []

        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
    except Exception as e:
        print(f"Error listing documentation pages: {e}")
        return []
    
@pydantic_ai_expert.tool
async def get_page_content(ctx:RunContext[PydanticAIDeps],url:str)->str:
    """
    Retrieve full contents of a specified documentation page by combining all its chunks.

    Args:
        ctx: The context including the Supabase client
        url: url of the page to retrieve

    Returns:
        str: The complete page content with all the chunks combined in order

    """
    try:
        results = ctx.deps.supabase.from_('site_pages')\
        .select('title,content,chunk_number')\
        .eq('url',url)\
        .eq('metadata->>source','PydanticAI_docs')\
        .order('chunk_number')\
        .execute()

        if not results.data:
            return f"No content found for url: {url}"
        
        page_title = results.data[0]['title'].split('-')[0]
        formatted_content = [f"#{page_title}\n"]

        #add each chunk content
        for chunk in results.data:
            formatted_content.append(chunk['content'])

        #create a clean format
        return "\n\n".join(formatted_content)
    except Exception as e:
        print(f"Error getting the page content: {e}")
        return f"Error getting the page content: {str(e)}"









