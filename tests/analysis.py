import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import sys
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. 경로 설정 (src를 찾기 위함)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MatrixFactorization
from src.dataset import MovieDataset

# 2. 데이터 및 모델 로드
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data = pd.read_csv(config['path']['data_path'])
n_users = data['user'].nunique()
n_movies = data['movie'].nunique()

X = data[['user', 'movie']].values
y = data['rating'].values

# 학습 때와 동일한 데이터 분할
_, X_test, _, y_test = train_test_split(
    X, y, test_size=config['train']['test_size'], random_state=config['train']['random_state']
)

# 모델 불러오기
model = MatrixFactorization(n_users, n_movies, embedding_dim=config['model']['embedding_dim'])
model.load_state_dict(torch.load(config['path']['model_path']))
model.eval()

# 3. 예측값 추출
with torch.no_grad():
    users = torch.LongTensor(X_test[:, 0])
    movies = torch.LongTensor(X_test[:, 1])
    all_preds = model(users, movies).numpy()

# 4. 시각화 함수 실행
def plot_analysis(y_test, all_preds):
    plt.figure(figsize=(14, 6))
    
    # [그래프 1] 수정된 버전: 두 데이터를 모두 히스토그램으로 겹쳐 그리기
    plt.subplot(1, 2, 1)
    
    # 실제 평점 (빨간색)
    sns.histplot(y_test, bins=5, color='red', alpha=0.3, label='Actual (Real)', kde=False)
    
    # 모델 예측값 (파란색) - kde=True를 여기에 넣으면 무조건 곡선이 나옵니다!
    sns.histplot(all_preds, bins=30, color='blue', alpha=0.5, label='Predicted (Model)', kde=True)
    
    plt.title('Rating Distribution: Reality vs Model')
    plt.xlabel('Rating Score')
    plt.xlim(0.5, 5.5) # 평점 범위 고정
    plt.legend()
    
    # [그래프 2] 오차 분포 (이건 동일하게)
    plt.subplot(1, 2, 2)
    errors = y_test - all_preds
    sns.histplot(errors, bins=30, kde=True, color='green')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title('How much the Model Missed (Error)')
    plt.xlabel('Error (Actual - Prediction)')
    
    plt.tight_layout()
    plt.show()

# 실행!
print("📊 데이터를 분석 중입니다... 잠시만 기다려주세요.")
plot_analysis(y_test, all_preds)