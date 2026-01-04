import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Foot Segmentation using SAM", layout="centered")

st.title("👣 Foot Segmentation using SAM")
st.write(
    "This web app visualizes foot segmentation results generated offline "
    "using the Segment Anything Model (SAM)."
)

st.markdown("---")

# Input image
st.subheader("Input Image")
input_path = "results/input_image.png"

if os.path.exists(input_path):
    img = Image.open(input_path)
    st.image(img, use_container_width=True)
else:
    st.warning("Input image not found in repository.")

st.markdown("---")

# Output image
st.subheader("Final Foot Segmentation Output")
output_path = "results/final_foot_segmentation.png"

if os.path.exists(output_path):
    result = Image.open(output_path)
    st.image(result, use_container_width=True)
    st.success("Foot segmentation displayed successfully!")
else:
    st.error("Segmentation result not found. Please upload results folder.")
