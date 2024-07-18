import subprocess
import yaml
import os
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import time

# Hyperopt 목표 함수 정의
def objective(params):
    # 하이퍼파라미터 YAML 파일 생성
    hyp_path = 'hyp.yaml'
    with open(hyp_path, 'w') as f:
        yaml.dump(params, f)

    # YOLOv5 학습 스크립트 호출
    cmd = [
        'python', 'train.py',
        '--epochs', '50',
        '--data', 'data/face.yaml',
        '--weights', 'yolov5s.pt',
        '--cache', 'True',
        '--save_period', '-1',
        '--batch-size', '16',
        '--cos-lr', 'True',
        '--hyp', hyp_path
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # runs/train 폴더 내의 최신 폴더를 기다림
    runs_dir = 'runs/train'
    latest_subdir = None
    while not latest_subdir:
        subdirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        if subdirs:
            latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)))
        else:
            print("Waiting for training directory to be created...")
            time.sleep(10)  # 10초 대기

    results_file = os.path.join(runs_dir, latest_subdir, 'results.csv')
    while not os.path.exists(results_file):
        print("Waiting for results file to be created...")
        time.sleep(10)  # 10초 대기

    # 학습 프로세스가 종료될 때까지 대기
    process.wait()
    
    try:
        with open(results_file) as f:
            lines = f.readlines()
            # 마지막 줄에서 mAP 값 추출
            last_line = lines[-1]
            mAP = float(last_line.split(',')[7])  # 필요에 따라 인덱스 조정
    except Exception as e:
        print(f"Error reading results: {e}")
        mAP = 0.0

    return {'loss': -mAP, 'status': STATUS_OK}

# 하이퍼파라미터 탐색 공간 정의
space = {
    'lr0': hp.uniform('lr0', 1e-4, 1e-2),
    'momentum': hp.uniform('momentum', 0.8, 0.99),
    'weight_decay': hp.uniform('weight_decay', 1e-5, 1e-3),
    'hsv_h': hp.uniform('hsv_h', 0.0, 0.1),
    'hsv_s': hp.uniform('hsv_s', 0.0, 0.9),
    'hsv_v': hp.uniform('hsv_v', 0.0, 0.9),
    'degrees': hp.uniform('degrees', 0.0, 45.0),
    'translate': hp.uniform('translate', 0.0, 0.1),
    'scale': hp.uniform('scale', 0.0, 0.5),
    'shear': hp.uniform('shear', 0.0, 10.0),
}

# Hyperopt 최적화
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials
)

print("Best hyperparameters:", best)
