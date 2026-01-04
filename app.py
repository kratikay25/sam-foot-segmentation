import streamlit as st
from PIL import Image
import os

st.set_page_config(
    page_title="Foot Segmentation using SAM",
    layout="centered"
)

st.title("👣 Foot Segmentation using SAM")
st.write(
    "This app demonstrates the **results** of foot segmentation using "
    "Meta AI's Segment Anything Model (SAM)."
)

st.divider()

st.subheader("📤 Upload Input Image")
uploaded = st.file_uploader(
    "Choose a foot image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Input Image", use_container_width=True)
    st.success("Image uploaded successfully!")

    st.divider()

    st.subheader("🧠 SAM Segmentation Output")

    result_path = "results/final_foot_segmentation.png"

    if os.path.exists(result_path):
        result = Image.open(result_path)
        st.image(
            result,
            caption="Final Foot Segmentation (SAM)",
            use_container_width=True
        )
        st.success("Hierarchical foot segmentation completed successfully!")
    else:
        st.error(
            "Segmentation result not found. "
            "Please ensure `results/final_foot_segmentation.png` exists."
        )
