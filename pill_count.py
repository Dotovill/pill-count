import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="필아이 스타일 카운터", layout="centered")
st.markdown("<h2 style='text-align: center;'>🟢 Pill Counter Live</h2>", unsafe_allow_html=True)

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 인식 정밀도")
    # 배경 대비 감도 (낮을수록 민감하게 반응)
    sens = st.slider("민감도", 10, 255, 120)
    # 잡음 제거 강도 (홀수만 가능)
    blur_val = st.slider("노이즈 제거", 1, 31, 15, step=2)
    # 너무 작은 점 무시
    min_area = st.slider("최소 알약 크기", 50, 3000, 300)

img_file = st.camera_input("알약 사진을 찍어주세요")

if img_file:
    img = Image.open(img_file)
    frame = np.array(img)
    
    # 1. 전처리 (흑백 -> 블러)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)
    
    # 2. 이진화 (배경과 알약 분리)
    _, thresh = cv2.threshold(blurred, sens, 255, cv2.THRESH_BINARY_INV)
    
    # 3. 테두리 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 결과를 그릴 빈 이미지 준비
    output_img = frame.copy()
    count = 0
    
    # ❗ [수정된 핵심 부분] 모든 테두리를 하나씩 돌며 점을 찍습니다.
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                # 알약의 중심점 계산
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # 초록색 점 찍기 (이 코드가 for문 안에서 매번 실행되어야 합니다)
                cv2.circle(output_img, (cX, cY), 20, (0, 255, 0), -1)
                # 하얀색 테두리 추가
                cv2.circle(output_img, (cX, cY), 25, (255, 255, 255), 3)
                count += 1

    # 4. 결과 출력
    st.image(output_img, use_container_width=True)
    
    # 하단 카운트 박스
    st.markdown(f"""
        <div style="background-color:#1e1e1e; padding:20px; border-radius:15px; text-align:center;">
            <span style="color:#00ff00; font-size:80px; font-weight:bold;">{count}</span>
            <p style="color:white; font-size:20px;">알약이 감지되었습니다</p>
        </div>
    """, unsafe_allow_html=True)
    
    # AI가 세상을 어떻게 보고 있는지 확인 (디버그용)
    with st.expander("인식 과정 확인"):
        st.image(thresh, caption="하얗게 뭉쳐 보이는 부분이 있다면 약 사이를 띄워주세요.")
