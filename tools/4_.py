# 犯罪數據庫

import streamlit as st
import boto3
import json
import base64
import io
from io import BytesIO
from PIL import Image
import pandas as pd
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
def call_claude_sonnet_text(retrieve_data, prompt):
    new_prompt = f"""
        Here are the search results:
        {retrieve_data}
        using traditional chinese
        """ + prompt
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [{"type": "text", "text": new_prompt}]}],
    }
    body = json.dumps(prompt_config)
    response = bedrock_runtime.invoke_model(body=body, modelId="anthropic.claude-3-5-sonnet-20240620-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0].get("text")


st.title("📝 犯罪數據庫")

col1, col2 = st.columns(2)

# display data
categories = ['全般刑法', '治安相關', '普通刑法', '賭博', '毒品', '交通事故', '酒駕']
df = pd.read_csv("data.csv")
last_month = df.iloc[-1, 1:]
last_year = df.iloc[-12:, 1:]

last_month_max_category = last_month.idxmax()
last_month_max_count = last_month.max().item()

last_year_count = last_year.iloc[-12][last_month_max_category].item()
delta = last_month_max_count - last_year_count

# 前一個月最大案件類別 + 前一個月最大案件類別跟去年該月相比
col1.metric("前一個月最大案件類別為", f'{last_month_max_category}, 共 {last_month_max_count} 件', delta=f'{delta} 件（相較去年同月）')


#
last_12_months = df.iloc[-12:, 1:]
previous_12_months = df.iloc[-24:-12, 1:]
percentage_changes = {}

for category in categories:
    percentage_changes[category] = (((last_12_months[category].sum() - previous_12_months[category].sum()) / previous_12_months[category].sum()) * 100).item()

col1.metric(f'整年上升比例最高案件類別', max(percentage_changes, key=percentage_changes.get), delta=max(list(percentage_changes.values())))

last_12_months_for_chart = df.iloc[-12:, :]
col2.bar_chart(last_12_months_for_chart[[last_month_max_category, '統計期']], y_label=f'{last_month_max_category} 件數', x_label='年分', x='統計期')




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

    # for data in retrieve_data["retrievalResults"]:
    #     print(f"Citation:\n{data}\n")

    text_output_from_claude = ""

    with st.spinner("Processing..."):
        text_output_from_claude = call_claude_sonnet_text(retrieve_data, prompt)

    st.session_state["session_4"]["messages"].append({"role": "assistant", "content": text_output_from_claude})
    st.chat_message("assistant").write(text_output_from_claude)