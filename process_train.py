# 필요한 라이브러리를 불러옵니다.
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Dropout

# 데이터 로딩 및 전처리 함수
def load_video_frames(video_path, frame_count=9, resize=(256, 256)):
    """비디오를 로드하고 3초 간격으로 프레임을 추출합니다."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = int(total_frames / frame_count)
    
    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백 변환
            frame = cv2.resize(frame, resize)  # 리사이즈
            frames.append(frame)
        else:
            break
    cap.release()
    return np.array(frames).reshape((frame_count, resize[0], resize[1], 1))

# 모델 정의 함수
def build_model(frame_count=9, frame_dim=256):
    """CNN + LSTM 모델을 정의합니다."""
    model = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(frame_count, frame_dim, frame_dim, 1)),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Conv2D(128, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        LSTM(256, return_sequences=False),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

# 모델 컴파일 및 학습, 평가 함수
def compile_and_train(model, X_train, y_train, X_val, y_val, epochs=10):
    """모델을 컴파일하고 학습, 평가를 수행합니다."""
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=10)
    return history

# 주석 처리된 코드: 아래의 코드는 함수들을 사용할 수 있도록 예제 데이터를 준비할 때 사용될 것입니다.
# # 예시: 비디오 경로가 주어졌을 때의 데이터 로딩
# video_path = 'path/to/video/file.mp4'
# frames = load_video_frames(video_path)

# # 모델 구축
# model = build_model()

# # 학습 및 평가
# # X_train, y_train, X_val, y_val 등은 실제 데이터셋으로부터 준비해야 합니다.
# history = compile_and_train(model, X_train, y_train, X_val, y_val)
