import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import yaml

from src.model import MatrixFactorization
from src.dataset import MovieDataset
from analysis import plot_analysis

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
    X, y, test_size=config['train']['test_size'], random_state=config['train']['random_state']
)

test_loader = DataLoader(MovieDataset(X_test, y_test), batch_size=config['train']['batch_size'], shuffle=False)

# 2. 모델 로드
model = MatrixFactorization(n_users, n_movies, embedding_dim=config['model']['embedding_dim'])
model.load_state_dict(torch.load(config['path']['model_path']))
model.eval()

# 3. 지표 계산 함수들
def get_metrics(model, X_test, y_test, k=10, threshold=4.0):
    all_preds = []
    all_targets = y_test
    
    # RMSE를 위한 예측값 추출
    with torch.no_grad():
        users = torch.LongTensor(X_test[:, 0])
        movies = torch.LongTensor(X_test[:, 1])
        preds = model(users, movies)
        all_preds = preds.numpy()

    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    # Precision@K & Recall@K 계산을 위해 데이터 정리
    # 유저별로 묶어서 계산해야 합니다.
    test_df = pd.DataFrame(X_test, columns=['user', 'movie'])
    test_df['actual'] = y_test
    test_df['pred'] = all_preds
    
    precisions = []
    recalls = []
    
    for user_id, group in test_df.groupby('user'):
        # 유저가 실제로 좋아한 영화 (기준 점수 이상)
        actual_liked = group[group['actual'] >= threshold]['movie'].tolist()
        if not actual_liked: continue # 좋아한 영화가 없으면 계산 제외
        
        # 모델이 예측한 상위 K개 영화
        top_k_recs = group.sort_values(by='pred', ascending=False).head(k)['movie'].tolist()
        
        # 맞힌 개수 (Intersection)
        hits = len(set(actual_liked) & set(top_k_recs))
        
        # Precision@K: 추천한 K개 중 맞힌 비율
        precisions.append(hits / k)
        # Recall@K: 유저가 좋아한 전체 중 맞힌 비율
        recalls.append(hits / len(actual_liked))
        
    return rmse, np.mean(precisions), np.mean(recalls)

# 4. 실행 및 출력
k_value = 10
rmse, precision, recall = get_metrics(model, X_test, y_test, k=k_value)

print("="*40)
print(f"🚀 모델 성능 평가 결과 (Top-{k_value})")
print(f"1. RMSE      : {rmse:.4f} (낮을수록 좋음)")
print(f"2. Precision : {precision*100:.2f}% (높을수록 좋음)")
print(f"3. Recall    : {recall*100:.2f}% (높을수록 좋음)")
print("="*40)
