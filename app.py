import streamlit as st
st.set_page_config(page_title="Skin Lesion Segmentation", layout="centered")

import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import requests

# --------------------------
# 下載模型架構與權重（如未存在）
# --------------------------
MODEL_URL = "https://huggingface.co/XiangtingLIN/unet-skin-lesion-model/resolve/main/unet_stat_attention_best.pth"
MODEL_FILE = "checkpoint/unet_stat_attention_best.pth"
MODEL_CODE_URL = "https://huggingface.co/XiangtingLIN/unet-skin-lesion-model/resolve/main/model.py"
MODEL_CODE_FILE = "model.py"

os.makedirs("checkpoint", exist_ok=True)

if not os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "wb") as f:
        f.write(requests.get(MODEL_URL).content)

if not os.path.exists(MODEL_CODE_FILE):
    with open(MODEL_CODE_FILE, "wb") as f:
        f.write(requests.get(MODEL_CODE_URL).content)

# 匯入模型架構
from model import UNet_EdgeBranch_AttentionGate

# --------------------------
# 載入模型
# --------------------------
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = UNet_EdgeBranch_AttentionGate()
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# --------------------------
# 預測用函式
# --------------------------
IMG_SIZE = 256
transform_img = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def predict(image: Image.Image):
    img_resized = image.resize((512, 512), Image.BILINEAR)
    img_tensor = transform_img(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        mask_logits, _ = model(img_tensor)
        pred_mask = (torch.sigmoid(mask_logits) > 0.5).float().cpu().numpy()[0, 0]

    pred_mask_img = Image.fromarray((pred_mask * 255).astype("uint8")).resize((512, 512), Image.NEAREST)
    return img_resized, pred_mask_img

# --------------------------
# Streamlit UI
# --------------------------
st.markdown("""
    <style>
        .main {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("功能選單")
option = st.sidebar.radio("選擇操作", ["上傳圖片", "模型預測"])

st.title("上傳皮膚病灶圖片")
uploaded_file = st.file_uploader("請上傳一張圖片（格式：jpg、png、bmp...）", type=["jpg", "png", "bmp", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="上傳的圖片", use_container_width=True)
    
    if st.button("開始預測"):
        with st.spinner("模型推論中..."):
            img_resized, result = predict(image)
        st.image(result, caption="預測mask", use_container_width=True)
        st.success("✅ 預測完成！")

        # ✅ 新增「加入紀錄並通知醫療單位」按鈕
        if st.button("📩 新增到我的紀錄並通知我的醫療單位"):
            # 模擬處理：儲存圖像、發送 API 請求或記錄動作
            st.success("✅ 已成功新增並通知您的醫療單位。")
