import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. 페이지 기본 설정
st.set_page_config(page_title="약사 전용 알약 카운터", layout="centered")
st.markdown("<h2 style='text-align: center;'>💊 약사 전용 알약 카운터</h2>", unsafe_allow_html=True)

# --- 핵심 분석 함수 ---
def count_pills(frame, sens, min_size, blur_val):
    # 1. 흑백 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 2. 노이즈 제거 (글씨처럼 얇은 선들을 뭉개서 지워버림)
    blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)
    
    # 3. 글씨 무시 로직: 배경보다 '밝은' 물체만 추출
    _, thresh = cv2.threshold(blurred, sens, 255, cv2.THRESH_BINARY)
    
    # 4. 테두리 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_img = frame.copy()
    count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_size:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # 초록색 점과 흰색 테두리 표시
                cv2.circle(output_img, (cX, cY), 20, (0, 255, 0), -1)
                cv2.circle(output_img, (cX, cY), 25, (255, 255, 255), 3)
                count += 1
                
    return output_img, count, thresh

# --- 사이드바: 조절 설정 ---
with st.sidebar:
    st.header("⚙️ 정밀 인식 설정")
    st.write("글씨가 같이 잡히면 '민감도'를 높이세요.")
    sens = st.slider("민감도 (밝기 기준)", 10, 255, 150)
    blur_val = st.slider("글씨 뭉개기 강도", 1, 31, 15, step=2)
    min_size = st.slider("최소 알약 크기", 100, 5000, 500)
    
    st.divider()
    st.info("💡 팁: 검은색 종이 위에서 찍으면 가장 정확합니다!")

# --- 메인 화면 탭 구성 ---
tab1, tab2 = st.tabs(["📸 카메라 촬영", "📁 파일 업로드"])

with tab1:
    img_snap = st.camera_input("알약을 평평하게 펴고 찍어주세요")
    if img_snap:
        img = Image.open(img_snap)
        frame = np.array(img)
        res_img, cnt, debug_img = count_pills(frame, sens, min_size, blur_val)
        st.image(res_img, use_container_width=True)
        st.success(f"감지된 알약: {cnt}개")
        with st.expander("AI 분석 레이어 확인"):
            st.image(debug_img, caption="흰색 덩어리만 알약으로 인식됩니다.")

with tab2:
    img_up = st.file_uploader("앨범에서 사진 선택", type=['jpg', 'jpeg', 'png'])
    if img_up:
        img = Image.open(img_up)
        frame = np.array(img)
        res_img, cnt, debug_img = count_pills(frame, sens, min_size, blur_val)
        st.image(res_img, use_container_width=True)
        st.metric("감지된 개수", f"{cnt} 개")
        with st.expander("AI 분석 레이어 확인"):
            st.image(debug_img, caption="분석용 흑백 화면")
