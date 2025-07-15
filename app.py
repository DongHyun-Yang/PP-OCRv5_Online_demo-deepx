import atexit
import base64
import io
import json
import os
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

import gradio as gr
import requests
from PIL import Image

API_URL = os.environ["API_URL"]

TOKEN = os.environ["API_TOKEN"]

TITLE = "PP-OCRv5 Online Demo"
DESCRIPTION = """
- PP-OCRv5 is the latest generation of the PP-OCR series model, designed to handle a wide range of scene and text types.
- It supports five major text types: Simplified Chinese, Traditional Chinese, Pinyin annotation, English, and Japanese.
- PP-OCRv5 has enhanced recognition capabilities for challenging use cases, including complex handwritten Chinese and English, vertical text, and rare characters.
- To use it, simply upload your image, or click one of the examples to load them. Read more at the links below.
"""

TEMP_DIR = tempfile.TemporaryDirectory()
atexit.register(TEMP_DIR.cleanup)

paddle_theme = gr.themes.Soft(
    font=["Roboto", "Open Sans", "Arial", "sans-serif"],
    font_mono=["Fira Code", "monospace"],
)
MAX_NUM_PAGES = 10
TMP_DELETE_TIME = 900
THREAD_WAKEUP_TIME = 600
CSS = """
:root {
    --sand-color: #FAF9F6;
    --white: #ffffff;
    --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    --text-color: #F3F4F7;
    --black:#000000;
    --link-hover: #2b6cb0;
    --content-width: 1200px;
}

body {
    display: flex;
    justify-content: center;
    background-color: var(--sand-color);
    color: var(--text-color);
    font-family: Arial, sans-serif;
}

.upload-section {
    width: 100%;
    margin: 0 auto 30px;
    padding: 20px;
    background-color: var(--sand-color) !important;
    border-radius: 8px;
    box-shadow: var(--shadow);
}

.center-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    margin-bottom: 20px;
}

.header {
    margin-bottom: 30px;
    width: 100%;
}

.logo-container {
    width: 100%;
    margin-bottom: 20px;
}

.logo-img {
    width: 100%;
    max-width: var(--content-width);
    margin: 0 auto;
    display: block;
}

.nav-bar {
    display: flex;
    justify-content: center;
    background-color: var(--white);
    padding: 15px 0;
    box-shadow: var(--shadow);
    margin-bottom: 20px;
}

.nav-links {
    display: flex;
    gap: 30px;
    width: 100%;
    justify-content: center;
}

.nav-link {
    color: var(--black);
    text-decoration: none;
    font-weight: bold;
    font-size: 24px;
    transition: color 0.2s;
}

.nav-link:hover {
    color: var(--link-hover);
    text-decoration: none;
}

button {
    background-color: var(--text-color) !important;
    color: var(--black) !important;
    border: none !important;
    border-radius: 4px;
    padding: 8px 16px;
}

.file-download {
    margin-top: 15px !important;
}
.loader {
    border: 5px solid #f3f3f3;
    border-top: 5px solid #3498db;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loader-container {
    text-align: center;
    margin: 20px 0;
}
.loader-container-prepare {
    text-align: left;
    margin: 20px 0;
}
.bold-label .gr-radio {
    margin-top: 8px;
    background-color: var(--white);
    padding: 10px;
    border-radius: 4px;
}

.bold-label .gr-radio label {
    font-size: 14px;
    color: var(--black);
}

#analyze-btn {
    background-color: #FF5722 !important;
    color: white !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
    position: fixed !important;
    bottom: 1% !important;
    left: 3% !important;
    z-index: 1000 !important;
}


#unzip-btn {
    background-color: #4CAF50 !important;
    color: white !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
    position: fixed !important;
    bottom: 1% !important;
    left: 18% !important;
    z-index: 1000 !important;
}

#download_file {
    position: fixed !important;
    bottom: 1% !important;
    left: 22% !important;
    z-index: 1000 !important;
}

#analyze-btn:hover,#unzip-btn:hover{
    transform: translateY(-3px) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
}

.square-pdf-btn {
    width: 90% !important;
    height: 3% !important;
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
}


.square-pdf-btn img {
    width: 20% !important;
    height: 1% !important;
    margin: 0 !important;
}


.square-pdf-btn span {
    font-size: 14px !important;
    text-align: center !important;
}


.gradio-gallery-item:hover {
    background-color: transparent !important;
    filter: none !important;
    transform: none !important;
}

.custom-markdown h3 {
    font-size: 25px !important;
}

.tight-spacing {
    margin-bottom: -20px !important;
}

.tight-spacing-as {
    margin-top: 0px !important;
    margin-bottom: 0px !important;
}

.left-margin-column {
    margin-left: 5%;
}

.image-container img {
    display: inline-block !important;
}

#markdown-title {
    text-align: center;
}


}
"""

EXAMPLE_TEST = [
    ["examples/ancient_demo.png"],
    ["examples/handwrite_ch_demo.png"],
    ["examples/handwrite_en_demo.png"],
    ["examples/japan_demo.png"],
    ["examples/magazine.png"],
    ["examples/pinyin_demo.png"],
    ["examples/research.png"],
    ["examples/tech.png"],
]
DESC_DICT = {
    "use_doc_orientation_classify": "Whether to use the document image orientation classification module. After use, you can correct distorted images, such as wrinkles, tilts, etc.",
    "use_doc_unwarping": "Whether to use the document unwarping module. After use, you can correct distorted images, such as wrinkles, tilts, etc.",
    "use_textline_orientation": "Whether to use the text line orientation classification module to support the distinction and correction of text lines of 0 degrees and 180 degrees.",
    "text_det_limit_type": "[Short side] means to ensure that the shortest side of the image is not less than [Image side length limit for text detection], and [Long side] means to ensure that the longest side of the image is not greater than [Image side length limit for text detection].",
    "text_det_limit_side_len_nb": "For the side length limit of the text detection input image, for large images with dense text, if you want more accurate recognition, you should choose a larger size. This parameter is used in conjunction with the [Image side length limit type for text detection]. Generally, the maximum [Long side] is suitable for scenes with large images and text, and the minimum [Short side] is suitable for document scenes with small and dense images.",
    "text_det_thresh_nb": "In the output probability map, only pixels with scores greater than the threshold are considered text pixels, and the value range is 0~1.",
    "text_det_box_thresh_nb": "When the average score of all pixels in the detection result border is greater than the threshold, the result will be considered as a text area, and the value range is 0 to 1. If missed detection occurs, this value can be appropriately lowered.",
    "text_det_unclip_ratio_nb": "Use this method to expand the text area. The larger the value, the larger the expanded area.",
    "text_rec_score_thresh_nb": "After text detection, the text box performs text recognition, and the text results with scores greater than the threshold will be retained. The value range is 0~1.",
}
tmp_time = {}
lock = threading.Lock()


def gen_tooltip_radio(desc_dict):
    tooltip = {}
    for key, desc in desc_dict.items():
        suffixes = ["_rd", "_md"]
        if key.endswith("_nb"):
            suffix = "_nb"
            suffixes = ["_nb", "_md"]
            key = key[: -len(suffix)]
        for suffix in suffixes:
            tooltip[f"{key}{suffix}"] = desc
    return tooltip


TOOLTIP_RADIO = gen_tooltip_radio(DESC_DICT)


def url_to_bytes(url, *, timeout=10):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def bytes_to_image(image_bytes):
    return Image.open(io.BytesIO(image_bytes))


def process_file(
    file_path,
    image_input,
    use_doc_orientation_classify,
    use_doc_unwarping,
    use_textline_orientation,
    text_det_limit_type,
    text_det_limit_side_len,
    text_det_thresh,
    text_det_box_thresh,
    text_det_unclip_ratio,
    text_rec_score_thresh,
):
    """Process uploaded file with API"""
    try:
        if not file_path and not image_input:
            raise ValueError("Please upload a file first")
        if file_path:
            if Path(file_path).suffix == ".pdf":
                file_type = "pdf"
            else:
                file_type = "image"
        else:
            file_path = image_input
            file_type = "image"
        # Read file content
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # Call API for processing

        file_data = base64.b64encode(file_bytes).decode("ascii")
        headers = {
            "Authorization": f"token {TOKEN}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            API_URL,
            json={
                "file": file_data,
                "fileType": 0 if file_type == "pdf" else 1,
                "useDocOrientationClassify": use_doc_orientation_classify,
                "useDocUnwarping": use_doc_unwarping,
                "useTextlineOrientation": use_textline_orientation,
                "textDetLimitType": text_det_limit_type,
                "textTetLimitSideLen": text_det_limit_side_len,
                "textDetThresh": text_det_thresh,
                "textDetBoxThresh": text_det_box_thresh,
                "textDetUnclipRatio": text_det_unclip_ratio,
                "textRecScoreThresh": text_rec_score_thresh,
            },
            headers=headers,
            timeout=1000,
        )
        try:
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError("API request failed") from e
        # Parse API response
        result = response.json()
        layout_results = result.get("result", {}).get("ocrResults", [])
        overall_ocr_res_images = []
        output_json = result.get("result", {})
        input_images = []
        input_images_gallery = []
        for res in layout_results:
            overall_ocr_res_images.append(url_to_bytes(res["ocrImage"]))
            input_images.append(url_to_bytes(res["inputImage"]))
            input_images_gallery.append(res["inputImage"])

        return {
            "original_file": file_path,
            "file_type": file_type,
            "overall_ocr_res_images": overall_ocr_res_images,
            "output_json": output_json,
            "input_images": input_images,
            "input_images_gallery": input_images_gallery,
            "api_response": result,
        }

    except requests.exceptions.RequestException as e:
        raise gr.Error(f"API request failed: {str(e)}")
    except Exception as e:
        raise gr.Error(f"Error processing file: {str(e)}")


def export_full_results(results):
    """Create ZIP file with all analysis results"""
    try:
        global tmp_time
        if not results:
            raise ValueError("No results to export")

        filename = Path(results["original_file"]).stem + f"_{uuid.uuid4().hex}.zip"
        zip_path = Path(TEMP_DIR.name, filename)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for i, img_bytes in enumerate(results["overall_ocr_res_images"]):
                zipf.writestr(f"overall_ocr_res_images/page_{i+1}.jpg", img_bytes)

            zipf.writestr(
                "output.json",
                json.dumps(results["output_json"], indent=2, ensure_ascii=False),
            )

            # Add API response
            api_response = results.get("api_response", {})
            zipf.writestr(
                "api_response.json",
                json.dumps(api_response, indent=2, ensure_ascii=False),
            )

            for i, img_bytes in enumerate(results["input_images"]):
                zipf.writestr(f"input_images/page_{i+1}.jpg", img_bytes)
        with lock:
            tmp_time[zip_path] = time.time()
        return str(zip_path)

    except Exception as e:
        raise gr.Error(f"Error creating ZIP file: {str(e)}")


def on_file_change(file):
    if file:
        return gr.Textbox(
            value=f"✅ Chosen file:  {os.path.basename(file.name)}", visible=True
        )
    else:
        return gr.Textbox()


def clear_file_selection():
    return gr.File(value=None), gr.Textbox(value=None)


def clear_file_selection_examples(image_input):
    text_name = "✅ Chosen file: " + os.path.basename(image_input)
    return gr.File(value=None), gr.Textbox(value=text_name, visible=True)


def toggle_sections(choice):
    return {
        Module_Options: gr.Column(visible=(choice == "Module Options")),
        Text_detection_Options: gr.Column(visible=(choice == "Text detection Options")),
    }


# Interaction logic
def toggle_spinner():
    return (
        gr.Column(visible=True),
        gr.Column(visible=False),
        gr.File(visible=False),
        gr.update(visible=False),
    )


def hide_spinner():
    return gr.Column(visible=False), gr.update(visible=True)


def update_display(results):
    if not results:
        return gr.skip()
    assert len(results["overall_ocr_res_images"]) <= MAX_NUM_PAGES, len(
        results["overall_ocr_res_images"]
    )
    assert len(results["input_images_gallery"]) <= MAX_NUM_PAGES, len(
        results["input_images_gallery"]
    )
    gallery_list_imgs = []
    for i in range(len(gallery_list)):
        gallery_list_imgs.append(
            gr.Gallery(
                value=results["input_images_gallery"],
                rows=len(results["input_images_gallery"]),
            )
        )
    ocr_imgs = []
    for img in results["overall_ocr_res_images"]:
        ocr_imgs.append(gr.Image(value=bytes_to_image(img), visible=True))
    for _ in range(len(results["overall_ocr_res_images"]), MAX_NUM_PAGES):
        ocr_imgs.append(gr.Image(visible=False))

    output_json = [gr.Markdown(value=results["output_json"], visible=True)]
    return ocr_imgs + output_json + gallery_list_imgs


def update_image(evt: gr.SelectData):
    update_images = []
    for index in range(MAX_NUM_PAGES):
        update_images.append(
            gr.Image(visible=False) if index != evt.index else gr.Image(visible=True)
        )
    return update_images


def delete_file_periodically():
    global tmp_time
    while True:
        current_time = time.time()
        delete_tmp = []
        for filename, strat_time in list(tmp_time.items()):
            if (current_time - strat_time) >= TMP_DELETE_TIME:
                if os.path.exists(filename):
                    os.remove(filename)
                    delete_tmp.append(filename)
        for filename in delete_tmp:
            with lock:
                del tmp_time[filename]
        time.sleep(THREAD_WAKEUP_TIME)


with gr.Blocks(css=CSS, title=TITLE, theme=paddle_theme) as demo:
    results_state = gr.State()
    gr.Markdown(
        value=f"# PP-OCRv5 Online Demo",
        elem_id="markdown-title",
    )
    gr.Markdown(value=DESCRIPTION)
    gr.Markdown(
        """
        Since our inference server is deployed in mainland China, cross-border
        network transmission may be slow, which could result in a suboptimal experience on Hugging Face.
        We recommend visiting the [PaddlePaddle AI Studio Community](https://aistudio.baidu.com/community/app/91660/webUI?source=appCenter) to try the demo for a smoother experience.
        """,
        elem_classes=["tight-spacing-as"],
        visible=True,
    )
    # Upload section
    with gr.Row():
        with gr.Column(scale=4):
            file_input = gr.File(
                label="Upload document",
                file_types=[".pdf", ".jpg", ".jpeg", ".png"],
                type="filepath",
                visible=False,
            )
            file_select = gr.Textbox(label="Select File Path", visible=False)
            image_input = gr.Image(
                label="Image",
                sources="upload",
                type="filepath",
                visible=False,
                interactive=True,
                placeholder="Click to upload image...",
            )
            pdf_btn = gr.Button(
                "Click to upload file...",
                variant="primary",
                icon="icon/upload.png",
                elem_classes=["square-pdf-btn"],
            )
            examples_image = gr.Examples(
                fn=clear_file_selection_examples,
                inputs=image_input,
                outputs=[file_input, file_select],
                examples_per_page=11,
                examples=EXAMPLE_TEST,
                run_on_click=True,
            )

            file_input.change(
                fn=on_file_change, inputs=file_input, outputs=[file_select]
            )
            with gr.Column():
                section_choice = gr.Dropdown(
                    choices=[
                        "Module Options",
                        "Text detection Options",
                    ],
                    value="Module Options",
                    label="Advance Options",
                    show_label=True,
                    container=True,
                    scale=0,
                    elem_classes=["tight-spacing"],
                )
                with gr.Column(
                    visible=True, elem_classes="left-margin-column"
                ) as Module_Options:
                    use_doc_orientation_classify_md = gr.Markdown(
                        "### Using the document image orientation classification module",
                        elem_id="use_doc_orientation_classify_md",
                    )
                    use_doc_orientation_classify_rd = gr.Radio(
                        choices=[("yes", True), ("no", False)],
                        value=False,
                        interactive=True,
                        show_label=False,
                        elem_id="use_doc_orientation_classify_rd",
                    )
                    use_doc_unwarping_md = gr.Markdown(
                        "### Using the document unwarping module",
                        elem_id="use_doc_unwarping_md",
                    )
                    use_doc_unwarping_rd = gr.Radio(
                        choices=[("yes", True), ("no", False)],
                        value=False,
                        interactive=True,
                        show_label=False,
                        elem_id="use_doc_unwarping_rd",
                    )
                    use_textline_orientation_md = gr.Markdown(
                        "### Using the text line orientation classification module",
                        elem_id="use_textline_orientation_md",
                    )
                    use_textline_orientation_rd = gr.Radio(
                        choices=[("yes", True), ("no", False)],
                        value=False,
                        interactive=True,
                        show_label=False,
                        elem_id="use_textline_orientation_rd",
                    )
                with gr.Column(
                    visible=False, elem_classes="left-margin-column"
                ) as Text_detection_Options:
                    text_det_limit_type_md = gr.Markdown(
                        "### Image side length restriction type for text detection",
                        elem_id="text_det_limit_type_md",
                    )
                    text_det_limit_type_rd = gr.Radio(
                        choices=[("Short side", "min"), ("Long side", "max")],
                        value="min",
                        interactive=True,
                        show_label=False,
                        elem_id="text_det_limit_type_rd",
                    )
                    text_det_limit_side_len_md = gr.Markdown(
                        "### Layout region detection expansion coefficient",
                        elem_id="text_det_limit_side_len_md",
                    )
                    text_det_limit_side_len_nb = gr.Number(
                        value=736,
                        step=1,
                        minimum=0,
                        maximum=10000,
                        interactive=True,
                        show_label=False,
                        elem_id="text_det_limit_side_len_nb",
                    )
                    text_det_thresh_md = gr.Markdown(
                        "### Text detection pixel threshold",
                        elem_id="text_det_thresh_md",
                    )
                    text_det_thresh_nb = gr.Number(
                        value=0.30,
                        step=0.01,
                        minimum=0.00,
                        maximum=1.00,
                        interactive=True,
                        show_label=False,
                        elem_id="text_det_thresh_nb",
                    )
                    text_det_box_thresh_md = gr.Markdown(
                        "### Text detection box threshold",
                        elem_id="text_det_box_thresh_md",
                    )
                    text_det_box_thresh_nb = gr.Number(
                        value=0.60,
                        step=0.01,
                        minimum=0.00,
                        maximum=1.00,
                        interactive=True,
                        show_label=False,
                        elem_id="text_det_box_thresh_nb",
                    )
                    text_det_unclip_ratio_md = gr.Markdown(
                        "### Text detection unclip ratio",
                        elem_id="text_det_unclip_ratio_md",
                    )
                    text_det_unclip_ratio_nb = gr.Number(
                        value=1.5,
                        step=0.1,
                        minimum=0,
                        maximum=10.0,
                        interactive=True,
                        show_label=False,
                        elem_id="text_det_unclip_ratio_nb",
                    )

                    text_rec_score_thresh_md = gr.Markdown(
                        "### Text recognition score threshold",
                        elem_id="text_rec_score_thresh_md",
                    )
                    text_rec_score_thresh_nb = gr.Number(
                        value=0.00,
                        step=0.01,
                        minimum=0,
                        maximum=1.00,
                        interactive=True,
                        show_label=False,
                        elem_id="text_rec_score_thresh_nb",
                    )
            with gr.Row():
                process_btn = gr.Button(
                    "🚀 Parse Document", elem_id="analyze-btn", variant="primary"
                )
                download_all_btn = gr.Button(
                    "📦 Download Full Results (ZIP)",
                    elem_id="unzip-btn",
                    variant="primary",
                )

        # Results display section
        with gr.Column(scale=7):
            gr.Markdown("### Results", elem_classes="custom-markdown")
            loading_spinner = gr.Column(
                visible=False, elem_classes=["loader-container"]
            )
            with loading_spinner:
                gr.HTML(
                    """
                <div class="loader"></div>
                <p>Processing, please wait...</p>
                """
                )
            prepare_spinner = gr.Column(
                visible=True, elem_classes=["loader-container-prepare"]
            )
            with prepare_spinner:
                gr.HTML(
                    """
                <div style="
                    max-width: 100%;
                    max-height: 100%;
                    margin: 24px 0 0 12px;
                    padding: 24px 32px;
                    border: 2px solid #A8C1E7;
                    border-radius: 12px;
                    background: #f8faff;
                    box-shadow: 0 2px 8px rgba(100,150,200,0.08);
                    font-size: 18px;
                ">
                    <b>🚀 User Guide</b><br>
                    <b>Step 1:</b> Upload Your File<br>
                    Supported formats: JPG, PNG, PDF, JPEG<br>
                    <b>Step 2:</b> Click Analyze Document Button<br>
                    System will process automatically<br>
                    <b>Step 3:</b> Wait for Results<br>
                    Results will be displayed after processing<br>
                    <b>Step 4:</b> Download results zip<br>
                    Results zip will be displayed after processing<br><br>
                    <b>Attention:</b> Only the first 10 pages will be processed
                </div>
                """
                )
            download_file = gr.File(visible=False, label="Download File")
            overall_ocr_res_images = []
            output_json_list = []
            gallery_list = []
            with gr.Tabs(visible=False) as tabs:
                with gr.Tab("OCR"):
                    with gr.Row():
                        with gr.Column(scale=2, min_width=1):
                            gallery_ocr_det = gr.Gallery(
                                show_label=False,
                                allow_preview=False,
                                preview=False,
                                columns=1,
                                min_width=10,
                                object_fit="contain",
                            )
                            gallery_list.append(gallery_ocr_det)
                        with gr.Column(scale=10):
                            for i in range(MAX_NUM_PAGES):
                                overall_ocr_res_images.append(
                                    gr.Image(
                                        label=f"OCR Image {i}",
                                        show_label=True,
                                        visible=False,
                                    )
                                )
                with gr.Tab("JSON"):
                    with gr.Row():
                        with gr.Column(scale=2, min_width=1):
                            gallery_json = gr.Gallery(
                                show_label=False,
                                allow_preview=False,
                                preview=False,
                                columns=1,
                                min_width=10,
                                object_fit="contain",
                            )
                            gallery_list.append(gallery_json)
                        with gr.Column(scale=10):
                            gr.HTML(
                                """
                            <style>
                            .line.svelte-19ir0ev svg {
                                width: 30px !important;
                                height: 30px !important;
                                min-width: 30px !important;
                                min-height: 30px !important;
                                padding: 0 !important;
                                font-size: 18px !important;
                            }
                            .line.svelte-19ir0ev span:contains('Object(') {
                                font-size: 12px;
                                }
                            </style>
                            """
                            )
                            output_json_list.append(
                                gr.JSON(
                                    visible=False,
                                )
                            )
    # # Navigation bar
    with gr.Column(elem_classes=["nav-bar"]):
        gr.HTML(
            """
        <div class="nav-links">
            <a href="https://github.com/PaddlePaddle/PaddleOCR" class="nav-link" target="_blank">GitHub</a>
        </div>
        """
        )

    section_choice.change(
        fn=toggle_sections,
        inputs=section_choice,
        outputs=[
            Module_Options,
            Text_detection_Options,
        ],
    )
    pdf_btn.click(
        fn=clear_file_selection, inputs=[], outputs=[file_input, file_select]
    ).then(
        None,
        [],
        [],
        js="""
        () => {
            const fileInput = document.querySelector('input[type="file"]');
            fileInput.value = '';
            fileInput.click();
        }
    """,
    )
    process_btn.click(
        toggle_spinner, outputs=[loading_spinner, prepare_spinner, download_file, tabs]
    ).then(
        process_file,
        inputs=[
            file_input,
            image_input,
            use_doc_orientation_classify_rd,
            use_doc_unwarping_rd,
            use_textline_orientation_rd,
            text_det_limit_type_rd,
            text_det_limit_side_len_nb,
            text_det_thresh_nb,
            text_det_box_thresh_nb,
            text_det_unclip_ratio_nb,
            text_rec_score_thresh_nb,
        ],
        outputs=[results_state],
    ).then(
        hide_spinner, outputs=[loading_spinner, tabs]
    ).then(
        update_display,
        inputs=[results_state],
        outputs=overall_ocr_res_images + output_json_list + gallery_list,
    )
    gallery_ocr_det.select(update_image, outputs=overall_ocr_res_images)

    download_all_btn.click(
        export_full_results, inputs=[results_state], outputs=[download_file]
    ).success(lambda: gr.File(visible=True), outputs=[download_file])

    demo.load(
        fn=lambda: None,
        inputs=[],
        outputs=[],
        js=f"""
        () => {{
            const tooltipTexts = {TOOLTIP_RADIO};
            let tooltip = document.getElementById("custom-tooltip");
            if (!tooltip) {{
                tooltip = document.createElement("div");
                tooltip.id = "custom-tooltip";
                tooltip.style.position = "fixed";
                tooltip.style.background = "rgba(0, 0, 0, 0.75)";
                tooltip.style.color = "white";
                tooltip.style.padding = "6px 10px";
                tooltip.style.borderRadius = "4px";
                tooltip.style.fontSize = "13px";
                tooltip.style.maxWidth = "300px";
                tooltip.style.zIndex = "10000";
                tooltip.style.pointerEvents = "none";
                tooltip.style.transition = "opacity 0.2s";
                tooltip.style.opacity = "0";
                tooltip.style.whiteSpace = "normal";
                document.body.appendChild(tooltip);
            }}
            Object.keys(tooltipTexts).forEach(id => {{
                const elem = document.getElementById(id);
                if (!elem) return;
                function showTooltip(e) {{
                    tooltip.style.opacity = "1";
                    tooltip.innerText = tooltipTexts[id];
                    let x = e.clientX + 10;
                    let y = e.clientY + 10;
                    if (x + tooltip.offsetWidth > window.innerWidth) {{
                        x = e.clientX - tooltip.offsetWidth - 10;
                    }}
                    if (y + tooltip.offsetHeight > window.innerHeight) {{
                        y = e.clientY - tooltip.offsetHeight - 10;
                    }}
                    tooltip.style.left = x + "px";
                    tooltip.style.top = y + "px";
                }}

                function hideTooltip() {{
                    tooltip.style.opacity = "0";
                }}

                elem.addEventListener("mousemove", showTooltip);
                elem.addEventListener("mouseleave", hideTooltip);
            }});
        }}
        """,
    )


if __name__ == "__main__":
    t = threading.Thread(target=delete_file_periodically)
    t.start()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
