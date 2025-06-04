import streamlit as st
from PIL import Image

st.set_page_config(page_title="Skin Lesion Segmentation", layout="centered")
st.title("🖼️ 上傳皮膚病灶圖片")

# 上傳圖片
uploaded_file = st.file_uploader("請上傳一張圖片（格式：jpg、png、bmp...）", type=["jpg", "png", "bmp", "jpeg"])

if uploaded_file is not None:
    # 顯示上傳的圖片
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="上傳的圖片", use_column_width=True)
    
    # 可選按鈕，之後可綁定模型推論功能
    if st.button("開始預測"):
        st.warning("⚠️ 尚未接上模型推論功能。")
