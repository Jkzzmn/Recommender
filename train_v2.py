import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
# 새로 정의한 모델 클래스명이 FeatureAidedGMF라면 아래와 같이 임포트하세요.
from src.model import FeatureAidedGMF 
from src.dataset import MovieDatasetV2

# 1. 설정 로드
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. 데이터 로드 및 분할
data = pd.read_csv(config['path']['data_v2_path'])
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# 3. Dataset 및 DataLoader 생성
train_dataset = MovieDatasetV2(train_df)
train_loader = DataLoader(train_dataset, batch_size=config['train_v2']['batch_size'], shuffle=True)

# 4. 모델 초기화 파라미터 계산
# 고정값 대신 데이터셋에 실제 생성된 피처 차원을 사용합니다.
n_users = data['user_id'].max()
n_movies = data['movie_id'].max()
genres_dim = train_dataset.genres_features.shape[1]
age_dim = train_dataset.age_features.shape[1]

model = FeatureAidedGMF(
    num_users=n_users, 
    num_items=n_movies, 
    embedding_dim=config['model_v2']['embedding_dim'], 
    genres_dim=genres_dim, 
    age_dim=age_dim
)

# 5. 학습 설정
epochs = config['train_v2']['epochs']
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config['train_v2']['learning_rate'])

# 6. 학습 루프
for epoch in range(epochs):
    model.train()
    total_loss = 0
    # 데이터셋의 __getitem__ 리턴 순서에 맞춰 5개를 받습니다.
    for u_ids, i_ids, g_feats, a_feats, ratings in train_loader:
        
        # 모델의 forward 구조에 맞게 4개의 인자를 전달합니다.
        preds = model(u_ids, i_ids, g_feats, a_feats)
        loss = criterion(preds, ratings.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# 7. 모델 저장
torch.save(model.state_dict(), config['path']['model_v2_path'])
print("🚀 모델 학습 및 저장 완료!")