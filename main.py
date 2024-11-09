from pdf2image import convert_from_path
import sys
sys.path.append('path/to/FastSAM')  # Adjust the path to where FastSAM is located
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from func import masks_to_bool, annotate_image
from concurrent.futures import ThreadPoolExecutor
from streamlit_drawable_canvas import st_canvas
import streamlit as st
import cv2, os, tempfile
import numpy as np
import supervision as sv
from PIL import Image
from ultralytics import YOLO

# Initialize models in session state only if not already loaded
if 'fast_sam' not in st.session_state:
    st.session_state.fast_sam = FastSAM("./model/FastSAM.pt")
if 'yolo_model' not in st.session_state:
    st.session_state.yolo_model = YOLO('./model/tile-detection.pt')


def pdf_to_images(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_pdf_path = tmp_file.name

    with ThreadPoolExecutor() as executor:
        images = executor.map(lambda page: convert_from_path(tmp_pdf_path, dpi=50, poppler_path='./poppler-0.68.0_x86/bin', first_page=page, last_page=page)[0], range(1, 1 + len(convert_from_path(tmp_pdf_path, dpi=50, poppler_path='./poppler-0.68.0_x86/bin'))))
    os.remove(tmp_pdf_path)
    st.success("PDF pages converted to images successfully!")
    return list(images)


def detect_tiles(img, area):
    results = st.session_state.fast_sam(
        source=img,
        device='cpu',
        retina_masks=True,
        imgsz=1024,
        conf=0.5,
        iou=0.6
    )
    prompt_process = FastSAMPrompt(img, results, device='cpu')
    masks = prompt_process.everything_prompt()
    masks = masks_to_bool(masks)
    xyxy = sv.mask_to_xyxy(masks=masks)
    fast_sam_detections = sv.Detections(xyxy=xyxy, mask=masks)

    result = st.session_state.yolo_model.predict(img, conf=0.50)[0].boxes.data
    detected_img = img.copy()

    bounding_box = []
    if area and len(result) > 0:
        for x, y, w, h in fast_sam_detections.xyxy:
            calc_area = (w - x) * (h - y)

            if area - 1500 <= calc_area <= area + 1200:
                bounding_box.append([x, y, w, h])
                cv2.rectangle(detected_img, (x, y), (w, h), (0, 255, 0), 2)

    return detected_img, bounding_box


# Main function
def main():
    st.set_page_config('Tile Detection', page_icon='📰', layout='wide')
    st.title('Tile Detection')
    st.sidebar.title("Navigation")

    # PDF Upload
    pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        # Check if a new file is uploaded
        if 'last_uploaded_pdf_name' not in st.session_state or st.session_state.last_uploaded_pdf_name != pdf_file.name:
            st.session_state.images = None
            st.session_state.idx = 0
            st.session_state.Cropped_image_Dic = {}
            st.session_state.last_uploaded_pdf_name = pdf_file.name

        # Convert PDF to images if not done already
        if 'images' not in st.session_state or st.session_state.images is None:
            with st.spinner("Converting PDF to images..."):
                st.session_state.images = pdf_to_images(pdf_file)
                st.session_state.Cropped_image_Dic = {idx: [] for idx in range(len(st.session_state.images))}

        # Step 1: Input area
        st.sidebar.subheader("Step 1: Enter Target Area")
        area = st.sidebar.number_input("Enter the target area size", min_value=0, step=1)
        st.session_state.area = area

        # Step 2: Annotation and Prediction
        idx = st.session_state.get('idx', 0)
        num_pages = len(st.session_state.images)

        st.subheader("Annotate a Tile to Save Area")
        annotated_img = Image.fromarray(np.array(st.session_state.images[idx]))

        # Canvas for annotation
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            background_image=annotated_img,
            update_streamlit=True,
            height=annotated_img.height,
            width=annotated_img.width,
            drawing_mode="rect",
            key="annotation_canvas",
        )

        # If a bounding box is drawn
        if canvas_result.json_data is not None:
            for shape in canvas_result.json_data["objects"]:
                if shape["type"] == "rect":
                    st.session_state.area = shape["width"] * shape["height"]
                    st.sidebar.success(f"Annotated area saved: {st.session_state.area}")
                    break

        if area > 0:
            detected_img, bounding_box = detect_tiles(np.array(st.session_state.images[idx]), st.session_state.area)
            st.subheader("Detection Tile In Image.")
            if bounding_box not in st.session_state.Cropped_image_Dic[idx] and len(bounding_box) > 0:
                st.session_state.Cropped_image_Dic[idx].append(bounding_box)
            st.image(detected_img, caption=f"Detected Page {idx + 1}", width=700)
            st.info(st.session_state.Cropped_image_Dic)
            # st.info(len(st.session_state.Cropped_image_Dic))

        # Page Navigation
        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button('Previous Page', disabled=(idx <= 0)):
                if idx > 0:
                    st.session_state.idx -= 1
                    st.experimental_rerun()

        with col2:
            if st.button('Next Page', disabled=(idx >= num_pages - 1)):
                if idx < num_pages - 1:
                    st.session_state.idx += 1
                    st.experimental_rerun()


        # ----------------------------------------------------------------------------------------------
        # Manual Annotation Section
        with st.expander("Manual Annotation"):
            st.sidebar.header("Manual Annotation")

            # Initialize annotation state
            if "annotation_mode" not in st.session_state:
                st.session_state.annotation_mode = False
                st.session_state.annotations = []

            # Button to start annotation mode
            if st.sidebar.button("Start Annotation"):
                st.session_state.annotation_mode = True

            # Display image and annotation options if in annotation mode
            if st.session_state.annotation_mode:
                pil_image = Image.fromarray(np.array(st.session_state.images[st.session_state.idx]))
                realtime_update = st.sidebar.checkbox("Update in realtime", True)

                # Create a canvas component
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    update_streamlit=realtime_update,
                    stroke_width=2,
                    height=pil_image.height,
                    width=pil_image.width,
                    background_image=pil_image,
                    drawing_mode="rect",
                    key="canvas",
                )

                if canvas_result.json_data is not None:
                    st.session_state.annotations.append(canvas_result.json_data["objects"])
                    st.json(canvas_result.json_data["objects"])


            # Button to end annotation mode
            if st.sidebar.button("End Annotation"):
                st.session_state.annotation_mode = False
                for annotation in st.session_state.annotations:
                    if len(annotation) > 0:
                        x = annotation[0]['left']
                        y = annotation[0]['top']
                        w = annotation[0]['width']
                        h = annotation[0]['height']
                        for key in st.session_state.Cropped_image_Dic.keys():
                            if [x, y, w, h] not in st.session_state.Cropped_image_Dic[key]:
                                st.session_state.Cropped_image_Dic[key].append([x, y, w, h])

                st.success("Annotation mode ended.")

if __name__ == "__main__":
    main()
