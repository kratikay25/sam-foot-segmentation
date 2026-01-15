import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import os
import urllib.request
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

st.set_page_config(
    page_title="Foot Segmentation using SAM",
    layout="wide"
)

# ---------------- HEADER ---------------- #
col_logo, col_title = st.columns([1, 8])

with col_logo:
    st.image("assets/logo.png", width=80)

with col_title:
    st.title("Foot Segmentation using SAM")
    st.write("Upload a foot image to run segmentation using SAM.")

# ---------------- FILE UPLOAD ---------------- #
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- AUTO-DOWNLOAD SAM MODEL ---------------- #
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MODEL_PATH = "models/sam_vit_b_01ec64.pth"

@st.cache_resource
def load_sam():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading SAM model (one-time, please wait)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
    sam.to(device="cpu")

    return SamAutomaticMaskGenerator(
        sam,
        min_mask_region_area=800
    )

# ---------------- MAIN PIPELINE ---------------- #
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    h, w, _ = image_np.shape

    st.subheader("Input Image")
    st.image(image, use_container_width=True)

    mask_generator = load_sam()
    masks = mask_generator.generate(image_np)

    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    foot_mask = np.zeros((h, w), dtype=np.uint8)

    masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    selected = 0

    for mask in masks:
        seg = mask["segmentation"].astype(np.uint8)
        ys, xs = np.where(seg == 1)
        if len(xs) == 0:
            continue

        cy = ys.mean()
        box_h = ys.max() - ys.min()
        box_w = xs.max() - xs.min()

        skin_pixels = image_hsv[seg == 1]
        skin_ratio = np.mean(
            (skin_pixels[:, 1] > 30) & (skin_pixels[:, 2] > 50)
        )

        if skin_ratio > 0.3 and cy > h * 0.45 and box_h > 0.6 * box_w:
            foot_mask[seg == 1] = 255
            selected += 1

        if selected == 2:
            break

    kernel = np.ones((7, 7), np.uint8)
    foot_mask = cv2.morphologyEx(foot_mask, cv2.MORPH_CLOSE, kernel)
    foot_mask = cv2.morphologyEx(foot_mask, cv2.MORPH_OPEN, kernel)

    result = cv2.bitwise_and(image_np, image_np, mask=foot_mask)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("SAM Mask")
        st.image(foot_mask, clamp=True)

    with col2:
        st.subheader("Final Segmentation")
        st.image(result, use_container_width=True)

    st.success("Segmentation completed successfully.")
