# 犯罪現場生成

import streamlit as st
import boto3
import replicate
import json
import base64
import io
from PIL import Image
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
REGION = os.getenv("REGION")

# List of Stable Diffusion Preset Styles
sd_presets = [
    "None",
    "3d-model",
    "analog-film",
    "anime",
    "cinematic",
    "comic-book",
    "digital-art",
    "enhance",
    "fantasy-art",
    "isometric",
    "line-art",
    "low-poly",
    "modeling-compound",
    "neon-punk",
    "origami",
    "photographic",
    "pixel-art",
    "tile-texture",
]

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

# Bedrock api call to stable diffusion
def generate_image_sd(text):
    """
    Purpose:
        Uses Bedrock API to generate an Image
    Args/Requests:
         text: Prompt
         style: style for image
    Return:
        image: base64 string of image
    """
    body = {
        "text_prompts": [{"text": text
                          }],
        "cfg_scale": 10,
        "seed": 0,
        "steps": 50,
        "style_preset": "photographic"
    }

    body = json.dumps(body)

    modelId = "stability.stable-diffusion-xl-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("artifacts")[0].get("base64")
    return results



def convert_base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return img


def call_claude_sonnet(text):
    prompt = f"""
        Please generate text similar to the following examples based on the text input:
        Example one:
        front shot, portrait photo of a cute 22 y.o woman, looks away, full lips, natural skin, skin moles, stormy weather, (cinematic, film grain:1.1)
        Example two:
        dark shot, front shot, photo of a 25 y.o Latino man, perfect eyes, natural skin, skin moles, looks at viewer, cinematic shot
        text input: {text}
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


st.title("💬 犯罪現場生成")
st.caption("描述犯罪現場")

if "session_2" not in st.session_state:
    st.session_state["session_2"] = {}
    st.session_state["session_2"]["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state["session_2"]["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():


    st.session_state["session_2"]["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    text_output_from_claude = call_claude_sonnet(prompt)

    print(text_output_from_claude)

    base64_output_from_sd = generate_image_sd(text_output_from_claude)
    img_output_from_sd = convert_base64_to_image(base64_output_from_sd)
    img_output_from_sd.save("image.png")
    st.image(img_output_from_sd)


    # msg = "this is the result image"
    # st.session_state.messages.append({"role": "assistant", "content": msg})
    # st.chat_message("assistant").write()