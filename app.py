import streamlit as st
import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import urllib.request

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Foot Segmentation using SAM", layout="centered")
st.title("🦶 Foot Segmentation using SAM")
st.write("Upload an image to segment the **foot region** using Segment Anything Model (SAM).")

# ---------------- LOAD SAM (AUTO DOWNLOAD + CACHE) ----------------
@st.cache_resource
def load_sam():
    CHECKPOINT = "sam_vit_b_01ec64.pth"

    if not os.path.exists(CHECKPOINT):
        st.warning("Downloading SAM model (first time only, ~375MB)...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        urllib.request.urlretrieve(url, CHECKPOINT)

    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to(device="cpu")

    return SamAutomaticMaskGenerator(
        sam,
        min_mask_region_area=800
    )

mask_generator = load_sam()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Foot Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Please upload an image to continue.")
    st.stop()

# ---------------- LOAD IMAGE ----------------
image = Image.open(uploaded_file).convert("RGB")
image_np = np.array(image)
h, w, _ = image_np.shape

st.subheader("📥 Input Image")
st.image(image_np, use_container_width=True)

# Convert to HSV for skin detection
image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

# ---------------- RUN SAM ----------------
with st.spinner("Running SAM segmentation..."):
    masks = mask_generator.generate(image_np)

# ---------------- FOOT MASK SELECTION ----------------
foot_mask = np.zeros((h, w), dtype=np.uint8)

# Sort masks by area
masks = sorted(masks, key=lambda x: x["area"], reverse=True)
selected = 0

for mask in masks:
    seg = mask["segmentation"].astype(np.uint8)

    ys, xs = np.where(seg == 1)
    if len(xs) == 0:
        continue

    cy = int(ys.mean())
    box_h = ys.max() - ys.min()
    box_w = xs.max() - xs.min()

    # Skin color check
    skin_pixels = image_hsv[seg == 1]
    skin_ratio = np.mean(
        (skin_pixels[:, 1] > 30) & (skin_pixels[:, 2] > 50)
    )

    if (
        skin_ratio > 0.3 and
        cy > h * 0.45 and cy < h * 0.85 and
        box_h > 0.6 * box_w
    ):
        foot_mask[seg == 1] = 255
        selected += 1

    if selected == 2:
        break

# ---------------- MASK POLISHING ----------------
kernel = np.ones((7, 7), np.uint8)
foot_mask = cv2.morphologyEx(foot_mask, cv2.MORPH_CLOSE, kernel)
foot_mask = cv2.morphologyEx(foot_mask, cv2.MORPH_OPEN, kernel)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foot_mask)
clean_mask = np.zeros_like(foot_mask)

if num_labels > 2:
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = np.argsort(areas)[-2:] + 1
    for lbl in largest:
        clean_mask[labels == lbl] = 255
    foot_mask = clean_mask

# ---------------- OUTPUT VISUALS ----------------
sam_mask_vis = np.zeros_like(image_np)
sam_mask_vis[foot_mask == 255] = [255, 255, 255]

final_output = cv2.bitwise_and(image_np, image_np, mask=foot_mask)

st.subheader("🧠 SAM Mask")
st.image(sam_mask_vis, use_container_width=True)

st.subheader("✅ Final Foot Segmentation")
st.image(final_output, use_container_width=True)

st.success("Foot segmentation completed successfully!")
