import base64
import io
import os

import gradio as gr
import requests
from PIL import Image

API_URL = "https://t7nd0cf3u89ck4bf.aistudio-hub.baidu.com/ocr"
TOKEN = os.getenv("API_TOKEN", "")


def inference(img):
    with io.BytesIO() as buffer:
        img.save(buffer, format="png")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    headers = {
        "Authorization": f"token {TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        API_URL,
        json={
            "file": img_base64,
            "fileType": 1,
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useTextlineOrientation": False,
        },
        headers=headers,
        timeout=1000,
    )
    response.raise_for_status()

    result = response.json()
    ocr_img_url = result["result"]["ocrResults"][0]["ocrImage"]

    response = requests.get(ocr_img_url, timeout=10)
    response.raise_for_status()
    ocr_img_base64 = Image.open(io.BytesIO(response.content))

    return ocr_img_base64, result["result"]["ocrResults"][0]["prunedResult"]


title = "PP-OCRv5 Online Demo"
description = """
- PP-OCRv5 is the latest generation of the PP-OCR series model, designed to handle a wide range of scene and text types.
- It supports five major text types: Simplified Chinese, Traditional Chinese, Chinese Pinyin, English, and Japanese.
- PP-OCRv5 has enhanced recognition capabilities for challenging use cases, including complex handwritten Chinese and English, vertical text, and rare characters.
- To use it, simply upload your image, or click one of the examples to load them. Read more at the links below.
- [Docs](https://paddlepaddle.github.io/PaddleOCR/), [Github Repository](https://github.com/PaddlePaddle/PaddleOCR).
"""

examples = [
    ["examples/ancient_demo.png"],
    ["examples/handwrite_ch_demo.png"],
    ["examples/handwrite_en_demo.png"],
    ["examples/japan_demo.png"],
    ["examples/magazine.png"],
    ["examples/pinyin_demo.png"],
    ["examples/research.png"],
    ["examples/tech.png"],
]

css = """
.output_image, .input_image {height: 40rem !important; width: 100% !important;}
h1 {text-align: center !important;}
"""

gr.Interface(
    inference,
    gr.Image(type="pil", label="Input Image"),
    [gr.Image(type="pil", label="Output Image"), gr.JSON(label="Output JSON", show_label=True)],
    title=title,
    description=description,
    examples=examples,
    cache_examples=False,
    css=css,
).launch(debug=False)
