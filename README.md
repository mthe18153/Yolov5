https://www.kaggle.com/datasets/iamtushara/face-detection-dataset/code  
캐글에서 데이터셋을 다운로드 받고
루트 디렉토리에 추가
데이터셋의 디렉토리 구조는  
input/face-data 안에 images,labels 폴더가 있는 시

input 폴더가 루트 디렉토리(yolov5-test 폴더)에 있으면 됨

train.py를 실행시키면 학습이 시작됨

별도의 터미널 명령어 입력 필요없이 바로 되도록 코드를 구성해놨음
if __name__ == "__main__":

    opt = parse_opt()
    opt.epochs = 50  # 총 학습 에포크 수를 50으로 설정
    opt.data = str(Path(__file__).resolve().parent / 'data' / 'face.yaml')  # 사용할 데이터셋 구성 파일 경로를 설정
    opt.weights = 'yolov5s.pt'  # 사전 학습된 가중치 파일 경로를 설정
    opt.cache = True  # 데이터셋을 캐시하여 학습 속도를 높임
    opt.save_period = -1  # 베스트 모델만 저장
    opt.batch_size = 16  # 배치 크기를 16으로 설정
    opt.cos_lr = True  # 코사인 학습률 스케줄러 활성화
    opt.amp = True  # Auto Mixed Precision Training 활성화
    
    main(opt)

이부분을 조정함으로써 여러가지 조작가능
