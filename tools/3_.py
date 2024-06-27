# 嫌犯人像生成

import streamlit as st
import boto3
import replicate
import json
import base64
import io
from PIL import Image
import os
from dotenv import load_dotenv
from streamlit_image_select import image_select

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
def generate_image_sd(text, seed):
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
        "seed": seed,
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

# function to convert PIL image to base64 string
def pil_to_base64(image, format="png"):
    with io.BytesIO() as buffer:
        image.save(buffer, format)
        return base64.b64encode(buffer.getvalue()).decode()

def convert_base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return img

def convert_pil_to_bytes(image, format='PNG'):
    # 將 PIL 圖像對象轉換成二進制格式
    byte_io = io.BytesIO()
    image.save(byte_io, format=format)
    byte_io.seek(0)  # 移動到流的開頭
    return byte_io.read()

def call_claude_sonnet(text):
    prompt = f"""
        Please generate text similar to the following examples based on the text input:
        Example one:
        A 30-year-old Asian man with a long, slender and oval face, looking gentle and handsome, a high and broad forehead, a square and full chin, narrow elongated eyes, a straight midsection of the face and thin lips, a long straight nose bridge with slight angular features, and an undercut hairstyle.
        Example two:
        Asian man, 50 years old, thick, jet-black hair, deep-set sharp eyes, pronounced double eyelids, thick bold eyebrows, bulbous nose,high nose bridge with large nostrils that are not visible. His face is wide and angular, resembling a square shape.
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


st.title("💬 嫌犯人像生成")
st.caption("描述嫌犯的長相")

if "session_3" not in st.session_state:
    st.session_state["session_3"] = {}
    st.session_state["session_3"]["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state["session_3"]["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():


    st.session_state["session_3"]["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    base64_output_from_sd = []
    img_output_from_sd = []
    columns = {}
    selected_image = None


    with st.spinner("Processing..."):
        if "picked_img" in st.session_state["session_3"]:
            init_img_b64 = pil_to_base64(st.session_state["session_3"]["picked_img"])
            updated_img = sd_update_image(change_prompt=st.session_state["session_3"]["last_prompt"] + prompt, init_image_b64=init_img_b64)
            updated_img = convert_base64_to_image(updated_img)
            st.session_state["session_3"]["img_list2"] = []
            st.session_state["session_3"]["img_list2"].append(st.session_state["session_3"]["picked_img"])
            st.session_state["session_3"]["img_list2"].append(updated_img)

        else:
            text_output_from_claude = call_claude_sonnet(prompt)
            print(text_output_from_claude)
            st.session_state["session_3"]["last_prompt"] = text_output_from_claude
            for i in range(5):
                base64_output_from_sd.append(generate_image_sd(text_output_from_claude, i))
                img_output_from_sd.append(convert_base64_to_image(base64_output_from_sd[i]))
                if "img_list" not in st.session_state["session_3"]:
                    st.session_state["session_3"]["img_list"] = []
                st.session_state["session_3"]["img_list"].append(img_output_from_sd[i])

if "img_list2" in st.session_state["session_3"]:
    selected_image2 = image_select("Select One", 
                                st.session_state["session_3"]["img_list2"],
                                use_container_width=False, key="unique_image_selector_2")
    st.session_state["session_3"]["picked_img"] = selected_image2
    # 轉換 PIL 圖像到適合下載的二進制數據
    if isinstance(selected_image2, Image.Image):  # 確認選中的對象是 PIL 圖像
        binary_data = convert_pil_to_bytes(selected_image2)
        st.download_button("Download image", binary_data, file_name="download.png")

elif "img_list" in st.session_state["session_3"]:
    selected_image = image_select("Select One", 
                                st.session_state["session_3"]["img_list"],
                                use_container_width=False, key="unique_image_selector_1")
    st.session_state["session_3"]["picked_img"] = selected_image



if "picked_img" in st.session_state["session_3"] and st.session_state["session_3"]["picked_img"]:
    st.image(st.session_state["session_3"]["picked_img"])
    # for i in range(5):
    #     st.image(img_output_from_sd[i])

    # st.session_state.messages.append({"role": "assistant", "content": st.image(img_output_from_sd)})
    # st.chat_message("assistant").write()