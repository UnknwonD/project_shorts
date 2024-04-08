import cv2

def compress_video(input_video_path, output_video_path, width=256, height=256, frame_rate=3):
    # 영상 읽기
    video_capture = cv2.VideoCapture(input_video_path)

    # 영상 전처리 후 저장을 위한 코드라인
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height), False)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # 영상 크기 조정
        resized_frame = cv2.resize(frame, (width, height))
        
        # 흑백 영상으로 변환
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        
        # 조정된 영상 쓰기
        out.write(gray_frame)
    
    # 종료 및 리소스 해제
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()




def compress_and_extract_highlights(input_video_path, output_video_path, highlights):
    video_capture = cv2.VideoCapture(input_video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 3초 간격의 프레임 번호 계산
    three_sec_frames = [int(fps * 3 * i) for i in range(frame_count // int(fps * 3))]
    
    # 하이라이트에 해당하는 프레임 번호 계산
    highlight_frames = [frame for i in highlights for frame in range(three_sec_frames[i], three_sec_frames[i] + int(fps * 3))]

    # 압축 및 하이라이트 추출
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 3, (256, 256), False)
    current_frame = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if current_frame in highlight_frames:
            resized_frame = cv2.resize(frame, (256, 256))
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            out.write(gray_frame)
        current_frame += 1

    video_capture.release()
    out.release()