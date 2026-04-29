import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="통합 알약 카운터", layout="centered")
st.title("💊 통합 알약 카운팅 시스템")

# --- 공통 분석 함수 ---
def count_pills(frame, sens, min_size, blur_val):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)
    _, thresh = cv2.threshold(blurred, sens, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_img = frame.copy()
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_size:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.circle(output_img, (cX, cY), 20, (0, 255, 0), -1)
                cv2.circle(output_img, (cX, cY), 25, (255, 255, 255), 3)
                count += 1
    return output_img, count

# --- 사이드바 설정 ---
with st.sidebar:
    st.header("⚙️ 인식 설정")
    sens = st.slider("민감도", 10, 255, 120)
    blur_val = st.slider("노이즈 제거", 1, 31, 15, step=2)
    min_size = st.slider("최소 알약 크기", 50, 3000, 300)

# --- 메인 화면 탭 (실시간 대신 2가지 강력한 모드) ---
tab1, tab2 = st.tabs(["📸 카메라 촬영", "📁 파일 업로드"])

with tab1:
    st.subheader("카메라로 즉시 분석")
    # 스마트폰 브라우저에서 가장 안정적인 방식
    img_snap = st.camera_input("알약을 비추고 사진을 찍으세요")
    if img_snap:
        img = Image.open(img_snap)
        res_img, cnt = count_pills(np.array(img), sens, min_size, blur_val)
        st.image(res_img)
        st.metric("감지된 개수", f"{cnt}개")

with tab2:
    st.subheader("앨범/파일에서 가져오기")
    img_up = st.file_uploader("이미지를 업로드하세요", type=['jpg','png','jpeg'])
    if img_up:
        img = Image.open(img_up)
        res_img, cnt = count_pills(np.array(img), sens, min_size, blur_val)
        st.image(res_img)
        st.metric("감지된 개수", f"{cnt}개")
