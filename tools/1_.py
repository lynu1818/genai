# 智能助手

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
from datetime import date

# Load the .env file
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
REGION = os.getenv("REGION")

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

# function to convert PIL image to base64 string
def pil_to_base64(image, format="png"):
    with io.BytesIO() as buffer:
        image.save(buffer, format)
        return base64.b64encode(buffer.getvalue()).decode()


def call_claude_sonnet_image(base64_string):
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_string,
                        },
                    },
                    {"type": "text", "text": "Please create a table and event description based on the photo that includes the following information: time of the event, location, type of incident, and vehicle number."},
                ],
            }
        ],
    }
    body = json.dumps(prompt_config)
    response = bedrock_runtime.invoke_model(body=body, modelId="anthropic.claude-3-5-sonnet-20240620-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0].get("text")

# knowledge base
def call_claude_sonnet(text, base64_string="", from_speech=False):
    content = []
    if base64_string != "":
        content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_string,
                        },
                    })
    
    today = date.today()

    if from_speech:
        prompt = f"""
            {text}
            Please create a table and event description based on the above description that includes the following information: time of the event, location, type of incident, and details.
            using traditional chinese
            using the following example markdown format:
            | 標題          | 內容                           |
            | -------------- | --------------------------------- |
            | 報案時間        | 11X年X月X日X時X分 (今日日期)        |
            | 發生時間        | 11X年X月X日X時X分 (輸入日期)        |
            | 地點           | 新竹市XX路                         |
            | 案件類型        | 一般刑案(妨礙自由、妨礙名譽(信用))、車(牌)輛協尋、違反社會秩序維護法、失蹤人口、身分不明、兒少性剝削擅離安置、遺失物、其他案類 |
            | 受理內容        | 民眾XXX,於11X年X月X日X時X分 (輸入日期),(輸入事件內容)，請求警方偵辦，依規定受理。 |
            
            today is {today}
            """
    elif text == "":
        prompt = f"""Please create a table and event description based on the photo that includes the following information: time of the event, location, type of incident, and vehicle number.
            using traditional chinese
            using the following example markdown format:
            | 標題          | 內容                          |
            | -------------- | --------------------------------- |
            | 發生時間        | 11X年X月X日X時X分 (今日日期)                          |
            | 發生地點       | No. 2 Zhonghua Road, Hsinchu City |
            | 案件類型      | 違規停車 or 交通事故 or 其他 違反「道路交通管理處罰條例第56條」, 在禁止臨時停車處所停車|
            | 車號         | BKK-3887                          |
            today is {today}
            """
    else:
        prompt = text

    content.append(
            {"type": "text", "text": prompt}
    )
    # print(prompt)
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {"role": "user", 
             "content": content}],
    }
    body = json.dumps(prompt_config)
    response = bedrock_runtime.invoke_model(body=body, modelId="anthropic.claude-3-5-sonnet-20240620-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0].get("text")


st.title("📝 智能助手")

if "session_1" not in st.session_state:
    st.session_state["session_1"] = {}
    st.session_state["session_1"]["messages"] = [{"role": "assistant", "content": "輸入圖片/拍張照片/錄音報案/文字輸入報案，我會協助處理日常案件！"}]


for msg in st.session_state["session_1"]["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


cols = {}
columns = st.columns(2)
for i in range(2):
    cols[f"col{i}"] = columns[i]
# Streamlit file uploader for only for images

uploaded_image = cols["col0"].file_uploader("上傳一張圖片", type=["png", "jpg", "jpeg"])
picture = cols["col1"].camera_input("拍一張照片")

if uploaded_image is not None:
    uploaded_image = Image.open(uploaded_image)
    st.image(uploaded_image)
    base64_string = pil_to_base64(uploaded_image)
    with st.spinner("處理中..."):
        text_output_from_claude = call_claude_sonnet("", base64_string)

        sound_file = BytesIO()
        tts = gTTS(text_output_from_claude, lang='zh', slow=False)
        tts.write_to_fp(sound_file)

    uploaded_image = None

    st.session_state["session_1"]["messages"].append({"role": "assistant", "content": text_output_from_claude})
    st.chat_message("assistant").write(text_output_from_claude)
    st.audio(sound_file)


if picture is not None:
    picture = Image.open(picture)
    st.image(picture)
    base64_string = pil_to_base64(picture)
    with st.spinner("處理中..."):
        text_output_from_claude = call_claude_sonnet("", base64_string)

        sound_file = BytesIO()
        tts = gTTS(text_output_from_claude, lang='zh', slow=False)
        tts.write_to_fp(sound_file)

    picture = None

    st.session_state["session_1"]["messages"].append({"role": "assistant", "content": text_output_from_claude})
    st.chat_message("assistant").write(text_output_from_claude)
    st.audio(sound_file)


speech_prompt = speech_to_text(language='zh-tw', key='my_stt', start_prompt="語音輸入", stop_prompt="停止錄音")


if prompt := st.chat_input("輸入訊息..."):
    st.session_state["session_1"]["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    text_output_from_claude = ""

    with st.spinner("處理中..."):
        text_output_from_claude = call_claude_sonnet(prompt, "")
    
    st.session_state["session_1"]["messages"].append({"role": "assistant", "content": text_output_from_claude})
    st.chat_message("assistant").write(text_output_from_claude)

elif speech_prompt:
    st.session_state["session_1"]["messages"].append({"role": "user", "content": speech_prompt})
    st.chat_message("user").write(speech_prompt)

    text_output_from_claude = ""

    with st.spinner("處理中..."):
        text_output_from_claude = call_claude_sonnet(speech_prompt, "", True)
    st.session_state["session_1"]["messages"].append({"role": "assistant", "content": text_output_from_claude})
    st.chat_message("assistant").write(text_output_from_claude)