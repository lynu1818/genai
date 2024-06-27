# 犯罪數據庫

import streamlit as st
import boto3
import json
import base64
import io
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS

# Load the .env file
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
REGION = os.getenv("REGION")
KB_ID = os.getenv("KB_ID")

# Initialize AWS clients
bedrock_agent_runtime = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


# knowledge base
def call_claude_sonnet_text(retrieve_data, question):
    prompt = f"""
        As a professional police officer, you have to predict the probability of crime based on the following data:
        {retrieve_data}
        given the data, answer the question: {question}
        """
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    body = json.dumps(prompt_config)
    response = bedrock_runtime.invoke_model(body=body, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0].get("text")


st.title("📝 犯罪數據庫")

if "session_4" not in st.session_state:
    st.session_state["session_4"] = {}
    st.session_state["session_4"]["messages"] = [{"role": "assistant", "content": "How can I help you?"}]


for msg in st.session_state["session_4"]["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])



if prompt := st.chat_input():
    
    st.session_state["session_4"]["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    retrieve_data = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": prompt},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 8}},
    )

    text_output_from_claude = ""

    with st.spinner("Processing..."):
        text_output_from_claude = call_claude_sonnet_text(retrieve_data, prompt)

    st.session_state["session_4"]["messages"].append({"role": "assistant", "content": text_output_from_claude})
    st.chat_message("assistant").write(text_output_from_claude)