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


def convert_base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return img


# function to convert PIL image to base64 string
def pil_to_base64(image, format="png"):
    with io.BytesIO() as buffer:
        image.save(buffer, format)
        return base64.b64encode(buffer.getvalue()).decode()
    
def convert_pil_to_bytes(image, format='PNG'):
    byte_io = io.BytesIO()
    image.save(byte_io, format=format)
    byte_io.seek(0)
    return byte_io.read()

def resize_image(image, size=(512, 512)):
    image = image.resize(size)
    return image

def sd_update_image(change_prompt, init_image_b64):
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
        "text_prompts": ([{"text": change_prompt, "weight": 1.0}]),
        "cfg_scale": 10,
        "init_image": init_image_b64,
        "seed": 0,
        "start_schedule": 0.6,
        "steps": 50,
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


st.title("💬 犯罪現場生成")
st.caption("描述犯罪現場")

if "session_2" not in st.session_state:
    st.session_state["session_2"] = {}
    st.session_state["session_2"]["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state["session_2"]["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# if uploaded_image is not None:
#     desc_image = ""
#     with st.spinner("Processing..."):
#         uploaded_image = Image.open(uploaded_image)
#         st.image(uploaded_image)
#         base64_string = pil_to_base64(uploaded_image)
#         desc_image = call_claude_sonnet_image(base64_string)
#     st.session_state["session_1"]["messages"].append({"role": "assistant", "content": desc_image})
#     st.chat_message("assistant").write(desc_image)
#     uploaded_image = None

updated_img = None

if prompt := st.chat_input():
    if uploaded_image is None:
        st.warning("Please upload an image.")
        
    else:
        st.session_state["session_2"]["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Processing..."):
            uploaded_image = Image.open(uploaded_image)
            resized_image = resize_image(uploaded_image)
            init_img_b64 = pil_to_base64(resized_image)
            updated_img = sd_update_image(prompt, init_image_b64=init_img_b64)
            updated_img = convert_base64_to_image(updated_img)
        st.image(updated_img)

if updated_img is not None and isinstance(updated_img, Image.Image):
    binary_data = convert_pil_to_bytes(updated_img)
    st.download_button("Download image", binary_data, file_name="download.png")