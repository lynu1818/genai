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
                    {"type": "text", "text": "Provide a caption for this image"},
                ],
            }
        ],
    }
    body = json.dumps(prompt_config)
    response = bedrock_runtime.invoke_model(body=body, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0].get("text")

# knowledge base
def call_claude_sonnet_text(text):
    prompt = f"""
        your are a professional police officer, answer the following question:
        {text}
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


# with st.sidebar:
#     anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
#     "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
#     "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("📝 全能小助手")



if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Streamlit file uploader for only for images
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    desc_image = ""
    with st.spinner("Processing..."):
        uploaded_image = Image.open(uploaded_image)
        st.image(uploaded_image)
        base64_string = pil_to_base64(uploaded_image)
        desc_image = call_claude_sonnet_image(base64_string)
    st.session_state.messages.append({"role": "assistant", "content": desc_image})
    st.chat_message("assistant").write(desc_image)
    uploaded_image = None

prompt = speech_to_text(key='my_stt', start_prompt="語音輸入", stop_prompt="停止錄音")

if prompt := st.chat_input() or prompt:
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    text_output_from_claude = ""
    sound_file = ""

    with st.spinner("Processing..."):
        text_output_from_claude = call_claude_sonnet_text(prompt)
        #語音輸出
        sound_file = BytesIO()
        tts = gTTS(text_output_from_claude, lang='zh', slow=False)
        tts.write_to_fp(sound_file)
    st.session_state.messages.append({"role": "assistant", "content": text_output_from_claude})
    st.chat_message("assistant").write(text_output_from_claude)
    st.audio(sound_file)