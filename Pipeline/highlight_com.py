import os
import numpy as np
import pandas as pd

from process_audio import extract_audio, preprocess_audio
from process_video import preprocess_video_every_3_seconds

from tensorflow.keras.models import load_model


import numpy as np

def ensemble_predictions(model1, model2, input_frame, input_audio):
    # 두 모델에서 예측값 가져오기
    preds1 = model1.predict(input_frame)  # 형상: (n_samples, 3)
    preds2 = model2.predict(input_audio)  # 형상: (n_samples, 2)
    
    # preds2를 preds1의 형상에 맞게 조정
    preds2_adjusted = np.zeros((preds2.shape[0], 3))
    preds2_adjusted[:, :2] = preds2  # 첫 두 열에 preds2 채우기
    preds2_adjusted[:, 2] = 1e-7     # 세 번째 열에 매우 작은 값 할당하여 확률이 0이 되지 않게 함

    # preds2_adjusted를 정규화하여 합이 1이 되도록 함
    preds2_adjusted /= preds2_adjusted.sum(axis=1, keepdims=True)

    # 예측값을 평균화하여 앙상블
    final_preds = (preds1 + preds2_adjusted) / 2

    return final_preds


def pipeline_video(video_path:str):
    ####### 1. 사용자가 입력한 비디오 불러오기
    if not os.path.exists(video_path):
        print(f"Video Not Found : {video_path}")
        return
    
    ####### 2. 사용자가 입력한 비디오 전처리하기
    # 1) 오디오와 비디오 분리
    # 2) 분리된 비디오와 오디오 각각 전처리
    
    audio = extract_audio(video_path, './test.wav')
    audio = preprocess_audio(audio)

    print("Audio Fin")

    video = preprocess_video_every_3_seconds(video_path, (256, 256), 3)

    print(len(video))
    # print(len(audio))
    ####### 3. 모델 불러오기
    # 1) 비디오 모델 load
    # 2) 오디오 모델 load

    video_model = load_model("video_3D_model.h5")
    # video_model = load_model("video_model.h5")
    audio_model = load_model("audio_comp_model.h5")

    ####### 4. 모델 추론
    # 1) 비디오 모델 Inference
    # 2) 오디오 모델 Inference

    video_output = video_model.predict(video)
    audio_output = audio_model.predict(audio)

    final_prediction = ensemble_predictions(video_model, audio_model, video, audio)

    ####### 5. Ensemble
    ##
    # ensemble_output = np.mean([video_output, audio_output], axis=0)
    print("##################audio######################")
    print(audio_output)
    print("#############################################")
    print("##################video######################")
    print(video_output)
    print("#############################################")
    # final_predictions = np.argmax(ensemble_output, axis=1)
    
    ####### 6. Return
    return final_prediction

video_path = "Pipeline/input.mp4"
test_data = pipeline_video(video_path)

print(test_data)