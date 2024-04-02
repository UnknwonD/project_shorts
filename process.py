import cv2

def resize_video(input_video_path, output_video_path, width=256, height=256):
    # 영상 읽기
    video_capture = cv2.VideoCapture(input_video_path)
    
    # 원본 영상 정보
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    # 영상 전처리 후 저장을 위한 코드라인
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 3, (width, height), False)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # 영상 크기 조정
        resized_frame = cv2.resize(frame, (width, height))
        
        # 조정된 영상 쓰기
        out.write(resized_frame)
    
    # 종료 및 리소스 해제
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = 'data\\원천데이터\\2~5분\\SUMVID_SHORT_TRAIN_01\\0a966c42cd1dc53e58eb2f7a4b675e6c8eca48dc44e545a28b35e1c5a61df61e-└»╞⌐║Ω ┐╡╗≤╣░(2~5║╨) └σ╕Θ║░ ┴▀┐Σ╡╡ ┼┬▒╫-100176-20201124135655-001-001.mp4'
    output_video_path = 'processed/resized_video_3_gray.mp4'

    resize_video(input_video_path, output_video_path)
