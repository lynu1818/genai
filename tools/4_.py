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
        As a data analyst, you've been given a dataset based on crime. You will conduct a detailed analysis of the current situation, covering the following aspects:
        Trend Analysis: Analyze the trend of crime data over time, identifying any significant upward or downward trends.
        Seasonal Analysis: Examine whether the crime data exhibits clear seasonal variations, such as increases or decreases in crime rates during specific seasons.
        Cyclic Analysis: Investigate if there are other cyclic patterns in the data, such as variations in crime frequency by week or month.
        ARIMA Model Prediction: Use the ARIMA model to forecast future crime data, providing estimated crime rates for the upcoming months.
        Here are the search results in numbered order:
        {retrieve_data}
        Given the data, answer the question: {question}
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
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 100}},
    )

    text_response = bedrock_agent_runtime.retrieve_and_generate(
        input = {
            "text": """ 
            As a data analyst, you've been given a dataset based on crime. You will conduct a detailed analysis of the current situation, covering the following aspects:
            
            Peaks and Troughs:
            Identify the highest and lowest crime data in the dataset along with their corresponding dates.
            Overall Trend:
            Describe the overall trend in crime data, noting any significant periodic fluctuations or changes in trend.
            Statistical Insights:
            Calculate and provide the average, median, and standard deviation of crime data.
            Anomalies:
            Point out any notable high or low outliers, including the dates of these anomalies.
            Chart Interpretation:
            Utilize generated data charts and provide interpretations based on these charts.
        
            """
        },
        retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": KB_ID,
            "modelArn": "anthropic.claude-3-sonnet-20240229-v1:0",
            },
        },
    )

    print(f"Output:\n{text_response['output']['text']}\n")

    # for data in retrieve_data["retrievalResults"]:
    #     print(f"Citation:\n{data}\n")

    text_output_from_claude = ""

    with st.spinner("Processing..."):
        text_output_from_claude = call_claude_sonnet_text(retrieve_data, prompt)

    st.session_state["session_4"]["messages"].append({"role": "assistant", "content": text_response['output']['text']})
    st.chat_message("assistant").write(text_response['output']['text'])