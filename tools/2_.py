# 犯罪現場生成

import streamlit as st
import boto3
import json
import base64
import io
from PIL import Image, ImageDraw
import os
from dotenv import load_dotenv
from streamlit_drawable_canvas import st_canvas
import pandas as pd

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


def convert_base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return img


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
    width, height = image.size

    ratio = min(size[0] / width, size[1] / height)

    new_width = int(width * ratio)
    new_height = int(height * ratio)

    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64

    if new_width == 0:
        new_width = 64
    if new_height == 0:
        new_height = 64

    resized_image = image.resize((new_width, new_height))

    return resized_image

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
                    {"type": "text", "text": "Please describe the image."},
                ],
            }
        ],
    }
    body = json.dumps(prompt_config)
    response = bedrock_runtime.invoke_model(body=body, modelId="anthropic.claude-3-5-sonnet-20240620-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get("body").read())
    print(f'response body: {response_body}')
    return response_body.get("content")[0].get("text")


def sd_inpaint_image(change_prompt, init_image_b64, mask):
    body = {
        "text_prompts": ([{"text": change_prompt, "weight": 1.0}]),
        "cfg_scale": 15,
        "init_image": init_image_b64,
        "mask_source": "MASK_IMAGE_WHITE",
        "mask_image": pil_to_base64(mask),
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



def sd_update_image(init_prompt, change_prompt, init_image_b64):
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
        "text_prompts": ([{"text": init_prompt + change_prompt, "weight": 1.0}]),
        "cfg_scale": 35,
        "init_image": init_image_b64,
        "image_strength": 0.5,
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

if "session_2" not in st.session_state:
    st.session_state["session_2"] = {}
    st.session_state["session_2"]["messages"] = [{"role": "assistant", "content": "輸入一張圖片，並給我犯罪顯場的描述，我會協助重現犯罪現場！"}]

for msg in st.session_state["session_2"]["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


uploaded_image_file = st.file_uploader("上傳一張圖片", type=["png", "jpg", "jpeg"])
uploaded_image = Image.open(uploaded_image_file) if uploaded_image_file else None

updated_img = None
mask_bg_image = None

if "last_image" in st.session_state["session_2"]:
    if "change_to_last_image" not in st.session_state["session_2"] or not st.session_state["session_2"]["change_to_last_image"]:
        st.session_state["session_2"]["change_to_last_image"] = True
        mask_bg_image = st.session_state["session_2"]["last_image"]
elif uploaded_image:
    mask_bg_image = uploaded_image
else:
    mask_bg_image = None


drawing_mode = st.selectbox(
    "Drawing tool:",
    ("rect", "circle", "transform", "polygon"),
)

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=3,
    background_image=mask_bg_image,
    height=512,
    width=512,
    drawing_mode=drawing_mode,
    point_display_radius=0,
    key="color_annotation_app",
)


# if "last_image" in st.session_state["session_2"]:
#     st.image(st.session_state["session_2"]["last_image"])



def draw_mask():
    mask_width, mask_height = 1024, 1024
    
    # Create a black background mask
    mask = Image.new("RGB", (mask_width, mask_height), color="black")
    draw = ImageDraw.Draw(mask)
    
    if canvas_result.json_data is not None:
        print("draw mask objects: ", canvas_result.json_data["objects"])
        df = pd.json_normalize(canvas_result.json_data["objects"])
        print(df)
        if len(df) == 0:
            return mask
        
        if drawing_mode == "rect":
            print("rectangle")
            # It's a rectangle
            top = df["top"] * 2
            left = df["left"] * 2
            rect_width = df["width"] * 2
            rect_height = df["height"] * 2
            
            # Draw the rectangle on the mask
            draw.rectangle([left, top, left + rect_width, top + rect_height], fill="#FFFFFF")
        elif drawing_mode == "polygon":
            print("polygon")
            paths = df["path"][0]
            print("path : ", paths)

            path_scaled = []
            # Scale the coordinates
            for path in paths:
                print(path)
                if len(path) == 3:
                    path_scaled.append([path[0], path[1]*2, path[2]*2])
            
            print("path_scaled: ", path_scaled)

            # Convert the path to a list of points
            polygon = [(point[1], point[2]) for point in path_scaled]
            print("polygon: ", polygon)

            # Draw the polygon on the mask
            draw.polygon(polygon, fill="#FFFFFF")
                
    mask.save("output_image.png")
    return mask

if prompt := st.chat_input("輸入訊息..."):
    if uploaded_image is None and "last_image" not in st.session_state["session_2"]:
        st.warning("請上傳一張圖片")
    
    elif "last_image" in st.session_state["session_2"]:
        st.session_state["session_2"]["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("處理中..."):
            init_img_b64 = pil_to_base64(st.session_state["session_2"]["last_image"])

            mask = draw_mask()
            inpainted_image = sd_inpaint_image(prompt, init_img_b64, mask)
            inpainted_image = convert_base64_to_image(inpainted_image)
        # st.image(inpainted_image)
        st.session_state["session_2"]["messages"].append({"role": "assistant", "content": inpainted_image})
        st.chat_message("assistant").write(inpainted_image)
        st.session_state["session_2"]["last_image"] = inpainted_image
        st.session_state["session_2"]["change_to_last_image"] = False
    else:
        st.session_state["session_2"]["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("處理中..."):
            # uploaded_image = Image.open(uploaded_image)
            # resized_image = resize_image(uploaded_image)

            init_img_b64 = pil_to_base64(uploaded_image)
            # init_img_b64 = pil_to_base64(resized_image)
            # text_output_from_claude = call_claude_sonnet_image(init_img_b64)
            # updated_img = sd_update_image(text_output_from_claude, prompt, init_image_b64=init_img_b64)
            # updated_img = convert_base64_to_image(updated_img)

            mask = draw_mask()
            canvas_result.json_data["objects"] = None
            print("objects: ", canvas_result.json_data["objects"])
            inpainted_image = sd_inpaint_image(prompt, init_img_b64, mask)
            inpainted_image = convert_base64_to_image(inpainted_image)
        # st.image(inpainted_image)
        st.session_state["session_2"]["last_image"] = inpainted_image
        st.session_state["session_2"]["messages"].append({"role": "assistant", "content": inpainted_image})
        st.chat_message("assistant").write(inpainted_image)
        # mask_bg_image = inpainted_image
        # st.image(updated_img)
        # st.session_state["session_2"]["messages"].append({"role": "assistant", "content": text_output_from_claude})
        # st.chat_message("assistant").write(text_output_from_claude)



if updated_img is not None and isinstance(updated_img, Image.Image):
    binary_data = convert_pil_to_bytes(updated_img)
    st.download_button("Download image", binary_data, file_name="download.png")



