import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. 화면 제목 설정
st.set_page_config(page_title="필아이 스타일 카운터")
st.title("🟢 실시간 알약 개수 카운터")
st.write("학습 데이터 없이 형태 분석으로만 개수를 셉니다.")

# 2. 사이드바 - 인식 조절 (약마다 크기가 다르니 여기서 조절)
st.sidebar.header("⚙️ 인식 정밀도")
sens = st.sidebar.slider("인식 민감도", 0, 255, 120)
min_size = st.sidebar.slider("최소 알약 크기", 100, 5000, 500)

# 3. 카메라 입력 (노트북 웹캠 연결)
img_file = st.camera_input("알약이 잘 보이게 카메라를 고정하고 찍어주세요")

if img_file:
    # 사진 데이터 읽기
    img = Image.open(img_file)
    frame = np.array(img)
    
    # --- 순수 이미지 처리 알고리즘 ---
    # 흑백으로 변환 (형태만 보기 위함)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # 이진화: 배경과 약을 0과 1로 분리 (민감도에 따라 결정)
    _, thresh = cv2.threshold(blurred, sens, 255, cv2.THRESH_BINARY)
    
    # 테두리(외곽선) 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_img = frame.copy()
    count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_size: # 너무 작은 먼지는 무시
            # 물체의 정중앙 좌표 구하기
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # 필아이처럼 초록색 점과 하얀 테두리 찍기
                cv2.circle(output_img, (cX, cY), 20, (0, 255, 0), -1)
                cv2.circle(output_img, (cX, cY), 25, (255, 255, 255), 3)
                count += 1

    # 4. 화면에 결과 표시
    st.image(output_img, use_container_width=True)
    
    # 큰 숫자로 개수 표시
    st.markdown(f"""
        <div style="text-align: center; background-color: #f0f2f6; padding: 20px; border-radius: 15px;">
            <p style="font-size: 24px; color: #31333F;">Total Count</p>
            <h1 style="font-size: 100px; color: #00FF00; margin: 0;">{count}</h1>
        </div>
    """, unsafe_allow_html=True)