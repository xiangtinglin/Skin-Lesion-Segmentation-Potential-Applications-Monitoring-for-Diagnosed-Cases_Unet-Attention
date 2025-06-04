import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from model import UNet_EdgeBranch_AttentionGate

@st.cache_resource
def load_model():
    model = UNet_EdgeBranch_AttentionGate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("./checkpoint/unet_stat_attention_best.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

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

    pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)
    return img_resized, pred_mask_img

st.set_page_config(page_title="Skin Lesion Segmentation", layout="centered")
st.title("🧠 Skin Lesion Segmentation (Apple Style)")
st.markdown("Upload a lesion image and get the segmented mask.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with st.spinner("Segmenting..."):
        img_resized, result = predict(image)

    with col2:
        st.subheader("Predicted Mask")
        st.image(result, use_column_width=True)

    st.success("Done! You can save the result if needed.")
