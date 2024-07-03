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
    body = {
        "text_prompts": [{"text": text}],
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
    byte_io = io.BytesIO()
    image.save(byte_io, format=format)
    byte_io.seek(0)
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
    response = bedrock_runtime.invoke_model(body=body, modelId="anthropic.claude-3-5-sonnet-20240620-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0].get("text")


st.title("💬 嫌犯人像生成")
# st.caption("描述嫌犯的長相")

if "session_3" not in st.session_state:
    st.session_state["session_3"] = {}
    st.session_state["session_3"]["messages"] = [{"role": "assistant", "content": "輸入文字描述，我將生成嫌疑犯的人像！"}]

for msg in st.session_state["session_3"]["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])



visible = st.toggle("顯示選項")
options = {"Age": "", "Ethnicity": "", "Gender": "", "Face Shape": "", "Eyes" : "", "Nose": "", "Mouth": "", "Hair Style": ""}

if visible:
    options["Age"] = st.selectbox("年齡", ["青少年 (1-18 歲)", "青壯年 (19-30 歲)", "中年 (31-64 歲)", "老年 (65-100 歲)", ""])
    options["Gender"] = st.selectbox("性別", ["男性", "女性", "其他", ""])
    options["Ethnicity"] = st.selectbox("種族", ["東亞人", "南亞人", "非洲人", "白人", "拉丁美洲人", ""])
    options["Hair Style"] = st.selectbox("髮型", ["短而直", "長而波浪", "中等長度卷曲", "平頭", "無髮", ""])
    options["Face Shape"] = st.selectbox("臉型", ["橢圓形", "方形", "圓形", "心形", "鑽石形", ""])
    options["Eyes"] = st.selectbox("眼睛", ["杏仁形", "圓形", "狹長形", "蒙古褶", "深陷型", ""])
    options["Nose"] = st.selectbox("鼻子", ["直型", "鷹勾型", "鈕扣型", "羅馬型", "寬型", ""])
    options["Mouth"] = st.selectbox("嘴巴", ["薄唇", "豐滿唇", "寬唇", "小嘴", "心形唇", ""])
    
if prompt := st.chat_input("輸入訊息..."):

    st.session_state["session_3"]["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    base64_output_from_sd = []
    img_output_from_sd = []
    columns = {}
    selected_image = None

    with st.spinner("處理中..."):
        if "picked_img" in st.session_state["session_3"]:
            init_img_b64 = pil_to_base64(st.session_state["session_3"]["picked_img"])
            updated_img = sd_update_image(change_prompt=st.session_state["session_3"]["last_prompt"] + prompt, init_image_b64=init_img_b64)
            updated_img = convert_base64_to_image(updated_img)
            st.session_state["session_3"]["img_list2"] = []
            st.session_state["session_3"]["img_list2"].append(st.session_state["session_3"]["picked_img"])
            st.session_state["session_3"]["img_list2"].append(updated_img)

        else:
            if visible:
                for option in options.values():
                    prompt += " " + option + " "
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

    if isinstance(selected_image2, Image.Image):
        binary_data = convert_pil_to_bytes(selected_image2)
        st.download_button("Download image", binary_data, file_name="download.png")

elif "img_list" in st.session_state["session_3"]:
    selected_image = image_select("Select One", 
                                st.session_state["session_3"]["img_list"],
                                use_container_width=False, key="unique_image_selector_1")
    st.session_state["session_3"]["picked_img"] = selected_image



if "picked_img" in st.session_state["session_3"] and st.session_state["session_3"]["picked_img"]:
    st.image(st.session_state["session_3"]["picked_img"])
