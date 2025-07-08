import os
import json
import sys
import logfire
import asyncio
import streamlit as st
from typing import Literal,TypedDict
from openai import AsyncOpenAI
from supabase import Client
from dotenv import load_dotenv

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelResponsePart
)

from agent import PydanticAIDeps,pydantic_ai_expert

#load environment variables
open_ai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

supabase : Client = Client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_KEY')

)

logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """format of the message to be sent to the browser/ API"""
    role:Literal['user','model']
    timestamp:str
    content:str

def display_message_part(part):
    """
    Display a  single part of a message in the streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    if part.part_kind =='system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**:{part.content}")
    
    #user-prompt
    elif part.part_kind =='user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)

    elif part.part_kind == 'text':
        with st.chat_message('assistant'):
            st.markdown(part.content)

async def  run_agent_with_streaming(user_input:str):
    """
    Run th agent with straming text for the user_input prompt,
    while maintaining the entire conversation in 'st.session_state.message'

    """
    #prepare dependencies
    deps = PydanticAIDeps(
        supabase=supabase,
        open_ai_client=open_ai_client
    )
    
    #run the agent with stream
    async with pydantic_ai_expert.run_stream(
        user_input,
        deps = deps,
        message_history = st.session_state.messages[:-1],
    ) as result:
        #we will gather text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        #render partial text as it arrives
        async for chunk in result.stream_text(delta= True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        #now we will filter the messages
        filtered_message = [msg for msg in result.new_messages()
                            if not (hasattr(msg,'parts') and
                                    any(part.part_kind == 'user_prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_message)

        #add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])

        )

async def main():
    st.title("Pydantic AI Expert")
    st.write("Ask any question about Pydantic AI, the hidden truths of the beauty of this framework lie within.")

    #initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg,ModelRequest) or isinstance(msg,ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    #chat input for the user
    user_input = st.chat_input("What question do you have about Pydantic AI?")

    if user_input:
        #We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        #display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        #display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            #Actual runt he agent now, streaming the text
            await run_agent_with_streaming(user_input)

if __name__ == "__main__":
    asyncio.run(main())

