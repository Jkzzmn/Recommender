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

from src.model import MatrixFactorization # 현재 사용 중인 NCF 모델
from src.dataset import MovieDataset

def run_integrated_evaluation():
    # 1. 설정 및 데이터 로드
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data = pd.read_csv(config['path']['data_path'])
    n_users = data['user'].nunique()
    n_movies = data['movie'].nunique()

    X = data[['user', 'movie']].values
    y = data['rating'].values

    # 학습 때와 동일한 split 유지
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=config['train']['test_size'], 
        random_state=config['train']['random_state']
    )

    # 2. 모델 로드 (v2 기준)
    model = MatrixFactorization(
        n_users, 
        n_movies, 
        embedding_dim=config['model']['embedding_dim']
    )
    model.load_state_dict(torch.load(config['path']['model_path']))
    model.eval()

    # 3. 예측값 추출 및 수치 지표 계산
    with torch.no_grad():
        users = torch.LongTensor(X_test[:, 0])
        movies = torch.LongTensor(X_test[:, 1])
        all_preds = model(users, movies).numpy()

    # RMSE 계산
    rmse = np.sqrt(mean_squared_error(y_test, all_preds))

    # Precision@K, Recall@K 계산
    k = 10
    threshold = 4.0
    test_df = pd.DataFrame(X_test, columns=['user', 'movie'])
    test_df['actual'] = y_test
    test_df['pred'] = all_preds
    
    precisions, recalls = [], []
    for _, group in test_df.groupby('user'):
        actual_liked = group[group['actual'] >= threshold]['movie'].tolist()
        if not actual_liked: continue
        
        top_k_recs = group.sort_values(by='pred', ascending=False).head(k)['movie'].tolist()
        hits = len(set(actual_liked) & set(top_k_recs))
        precisions.append(hits / k)
        recalls.append(hits / len(actual_liked))

    # 4. 결과 출력
    print("="*40)
    print(f"🚀 Matrix Factorization 결과 (Top-{k})")
    print(f"1. RMSE      : {rmse:.4f}")
    print(f"2. Precision : {np.mean(precisions)*100:.2f}%")
    print(f"3. Recall    : {np.mean(recalls)*100:.2f}%")
    print("="*40)

    # 5. 시각화 (Analysis 기능)
    plt.figure(figsize=(14, 6))
    
    # [왼쪽] 분포 비교
    plt.subplot(1, 2, 1)
    sns.histplot(y_test, bins=5, color='red', alpha=0.3, label='Actual', kde=False)
    sns.histplot(all_preds, bins=30, color='blue', alpha=0.5, label='Predicted', kde=True)
    plt.title('Rating Distribution')
    plt.legend()
    
    # [오른쪽] 오차 분포
    plt.subplot(1, 2, 2)
    errors = y_test - all_preds
    sns.histplot(errors, bins=30, kde=True, color='green')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title('Prediction Error Distribution')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_integrated_evaluation()