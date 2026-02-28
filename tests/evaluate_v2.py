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

# 1. 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 모델 클래스명을 FeatureAidedGMF로 바꾸셨다면 아래를 수정하세요.
from src.model import FeatureAidedGMF 

def run_evaluation():
    # 설정 로드
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 데이터 로드 및 전처리 (학습과 동일하게 나이/장르만 남김)
    data = pd.read_csv(config['path']['data_v2_path'])
    
    # 불필요한 성별, 직업 제거 및 bool 변환 (학습 코드와 일치시켜야 함)
    for col in ['gender', 'occupation']:
        if col in data.columns:
            data = data.drop(columns=[col])
            
    bool_cols = data.select_dtypes(include=['bool']).columns
    data[bool_cols] = data[bool_cols].astype(float)

    # 차원 계산
    n_users = data['user_id'].max()
    n_movies = data['movie_id'].max()
    
    age_cols = [c for c in data.columns if 'age_' in c]
    genres_cols = [c for c in data.columns if c not in ['user_id', 'movie_id', 'rating'] + age_cols]
    
    age_dim = len(age_cols)
    genres_dim = len(genres_cols)

    # 데이터 분할 (학습과 동일한 random_state)
    _, test_df = train_test_split(data, test_size=0.2, random_state=42)

    # 2. 모델 로드
    model = FeatureAidedGMF(
        num_users=n_users, 
        num_items=n_movies, 
        embedding_dim=config['model_v2']['embedding_dim'], 
        genres_dim=genres_dim, 
        age_dim=age_dim
    )
    model.load_state_dict(torch.load(config['path']['model_v2_path']))
    model.eval()

    # 3. 예측
    with torch.no_grad():
        u_ids = torch.LongTensor(test_df['user_id'].values)
        i_ids = torch.LongTensor(test_df['movie_id'].values)
        g_feats = torch.FloatTensor(test_df[genres_cols].values)
        a_feats = torch.FloatTensor(test_df[age_cols].values)
        
        # 4개의 인자 전달
        preds = model(u_ids, i_ids, g_feats, a_feats).numpy()

    test_df['pred'] = preds
    y_true = test_df['rating'].values

    # 4. 지표 계산
    rmse = np.sqrt(mean_squared_error(y_true, preds))

    K = 10
    threshold = 4.0
    precisions, recalls = [], []

    for user_id, group in test_df.groupby('user_id'):
        actual_liked = group[group['rating'] >= threshold]['movie_id'].tolist()
        if len(actual_liked) == 0: continue 

        top_k_recs = group.sort_values(by='pred', ascending=False).head(K)['movie_id'].tolist()
        hits = len(set(actual_liked) & set(top_k_recs))
        
        precisions.append(hits / K)
        recalls.append(hits / len(actual_liked))

    # 결과 출력
    print("="*40)
    print(f"🚀 Feature-Aided GMF 평가 결과 (Top-{K})")
    print(f"1. RMSE      : {rmse:.4f}")
    print(f"2. Precision : {np.mean(precisions)*100:.2f}%")
    print(f"3. Recall    : {np.mean(recalls)*100:.2f}%")
    print("="*40)

    # 5. 시각화 (기존 디자인 유지)
    plt.figure(figsize=(14, 6))
    
    # [왼쪽] 분포 비교
    plt.subplot(1, 2, 1)
    sns.histplot(y_true, bins=5, color='red', alpha=0.3, label='Actual', 
                 kde=False, stat='probability', element='bars')
    sns.histplot(preds, bins=30, color='blue', alpha=0.5, label='Predicted', 
                 kde=True, stat='probability', element='bars')
    plt.title('Rating Distribution (GMF v2)')
    plt.xlabel('Rating')
    plt.ylabel('Probability')
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend()
    
    # [오른쪽] 오차 분포
    plt.subplot(1, 2, 2)
    errors = y_true - preds
    sns.histplot(errors, bins=30, kde=True, color='green')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error (Actual - Predicted)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_evaluation()