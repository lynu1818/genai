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
    response = bedrock_runtime.invoke_model(body=body, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0].get("text")

# knowledge base
def call_claude_sonnet(text, base64_string=""):
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
    content.append(
        {"type": "text", "text": prompt}
    )
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {"role": "user", 
             "content": content}],
    }
    body = json.dumps(prompt_config)
    response = bedrock_runtime.invoke_model(body=body, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0].get("text")


st.title("📝 智能助手")

if "session_1" not in st.session_state:
    st.session_state["session_1"] = {}
    st.session_state["session_1"]["messages"] = [{"role": "assistant", "content": "How can I help you?"}]


for msg in st.session_state["session_1"]["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


cols = {}
columns = st.columns(2)
for i in range(2):
    cols[f"col{i}"] = columns[i]
# Streamlit file uploader for only for images

uploaded_image = cols["col0"].file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
picture = cols["col1"].camera_input("Take a picture")

prompt = speech_to_text(key='my_stt', start_prompt="語音輸入", stop_prompt="停止錄音")


if prompt := st.chat_input() or prompt:
    
    st.session_state["session_1"]["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    text_output_from_claude = ""
    sound_file = ""

    with st.spinner("Processing..."):
        if uploaded_image is not None:
            uploaded_image = Image.open(uploaded_image)
            st.image(uploaded_image)
            base64_string = pil_to_base64(uploaded_image)
            text_output_from_claude = call_claude_sonnet(prompt, base64_string)
        elif picture is not None:
            picture = Image.open(picture)
            st.image(picture)
            base64_string = pil_to_base64(picture)
            text_output_from_claude = call_claude_sonnet(prompt, base64_string)
        else:
            text_output_from_claude = call_claude_sonnet(prompt)
        #語音輸出
        sound_file = BytesIO()
        tts = gTTS(text_output_from_claude, lang='zh', slow=False)
        tts.write_to_fp(sound_file)
    uploaded_image = None
    picture = None
    st.session_state["session_1"]["messages"].append({"role": "assistant", "content": text_output_from_claude})
    st.chat_message("assistant").write(text_output_from_claude)
    st.audio(sound_file)