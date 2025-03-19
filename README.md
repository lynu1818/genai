# 警囊妙計（A Tailored AI-Integrated Platform for Police）
> Developed for the [2024 GenAI Stars Hackathon](https://genaistars.org.tw/award/hackathon/13)  
> Demo Video: [YouTube Demo](https://www.youtube.com/watch?v=mFWuRH6CZG8&t=2s)

## Overview

**警囊妙計** is an AI-powered platform that leverages **Amazon Bedrock** and advanced generative models to offer intelligent assistance across a range of tasks.

### Key Features

1. **Suspect Sketch Generation**  
   Automatically generates realistic suspect images based on descriptions.

2. **Crime Scene Reconstruction**  
   Visually reconstruct crime scenes based on textual descriptions and user-provided masks.

3. **Custom Crime Database & Forecasting**  
   Maintains a searchable database and performs crime trend prediction using past incident data.

4. **AI Case Assistant**  
   Helps officers process daily tasks and inquiries more efficiently.


## Installation and Setup


1. Clone the repository and install dependencies:
    ```bash
    git clone https://github.com/lynu1818/genai.git
    cd genai
    pip install -r requirements.txt
    ```

2. Create a .env file in the root directory and configure the following environment variables:
    ```bash
    REPLICATE_API_TOKEN=your_replicate_token
    AWS_ACCESS_KEY_ID=your_aws_key
    AWS_SECRET_ACCESS_KEY=your_aws_secret
    REGION=your_aws_region
    KB_ID=your_bedrock_knowledge_base_id
    ```

3. Launch the application:
    ```
    streamlit run app.py
    ```
4. Open the local URL provided by Streamlit in your browser to access the app