import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. 화면 설정
st.set_page_config(page_title="알약 카운터", layout="centered")
st.title("🟢 순수 이미지 분석 카운터")
st.write("이미지 분석을 통해 알약의 개수만 정확히 셉니다.")

# 2. 사이드바 설정 (인식 조절용)
with st.sidebar:
    st.header("⚙️ 인식 조절")
    # 배경이 어두울수록 인식이 잘 됩니다.
    sens = st.slider("민감도", 0, 255, 120)
    # 너무 작은 먼지나 글씨는 무시하도록 설정
    min_size = st.slider("최소 알약 크기", 100, 5000, 500)

# 3. 파일 업로드 (기본 카메라 앱 활용 가능)
img_file = st.file_uploader("알약 사진을 올리거나 새로 찍으세요", type=['jpg', 'jpeg', 'png'])

if img_file:
    # 사진 가져오기
    img = Image.open(img_file)
    frame = np.array(img)
    
    # --- 이미지 처리 과정 ---
    # A. 흑백 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # B. 노이즈 제거 (화질 개선)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # C. 이진화 (배경과 약 분리)
    _, thresh = cv2.threshold(blurred, sens, 255, cv2.THRESH_BINARY_INV)
    
    # D. 테두리 인식
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    display_img = frame.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_size: # 너무 작은 건 무시
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # 초록색 점 찍기
                cv2.circle(display_img, (cX, cY), 20, (0, 255, 0), -1)
                cv2.circle(display_img, (cX, cY), 25, (255, 255, 255), 3)
                count += 1

    # 4. 최종 결과 출력
    st.image(display_img, use_container_width=True)
    
    # 결과 숫자 표시
    st.markdown(f"""
        <div style="text-align: center; background-color: #262730; padding: 20px; border-radius: 10px;">
            <p style="font-size: 20px; color: white;">감지된 알약 수</p>
            <h1 style="font-size: 100px; color: #00FF00; margin: 0;">{count}</h1>
        </div>
    """, unsafe_allow_html=True)

    # 인식 상태 확인용 (디버깅)
    with st.expander("인식 과정 보기"):
        st.image(thresh, caption="흰색 부분이 약으로 인식된 영역입니다.")
