import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# 페이지 설정
st.set_page_config(page_title="통합 알약 카운터", layout="centered")
st.title("💊 통합 알약 카운팅 시스템")

# --- 공통 분석 함수 (AI 학습 데이터 없이 형태 분석) ---
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

# --- 메인 화면 탭 구성 ---
tab1, tab2, tab3 = st.tabs(["🎥 실시간 비추기", "📸 사진 찍기", "📁 파일 올리기"])

# 1) 실시간으로 개수 표시 (Live)
with tab1:
    st.subheader("실시간 라이브 카운팅")
    st.info("카메라를 비추면 실시간으로 점이 찍힙니다. (PC 환경 권장)")
    
    class PillProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="rgb24")
            processed_img, _ = count_pills(img, sens, min_size, blur_val)
            return frame.from_ndarray(processed_img, format="rgb24")

    webrtc_streamer(key="pill-live", video_processor_factory=PillProcessor)

# 2) 사진으로 개수 표시 (기존 촬영)
with tab2:
    st.subheader("카메라 촬영 분석")
    img_snap = st.camera_input("알약 사진을 찍으세요", key="snap")
    if img_snap:
        img = Image.open(img_snap)
        frame = np.array(img)
        res_img, cnt = count_pills(frame, sens, min_size, blur_val)
        st.image(res_img)
        st.success(f"감지된 개수: {cnt}개")

# 3) 파일 올려서 세기 (업로드)
with tab3:
    st.subheader("이미지 파일 업로드")
    img_up = st.file_uploader("이미지를 업로드하세요", type=['jpg','png','jpeg'], key="up")
    if img_up:
        img = Image.open(img_up)
        frame = np.array(img)
        res_img, cnt = count_pills(frame, sens, min_size, blur_val)
        st.image(res_img)
        st.metric("Total Count", f"{cnt} 개")
