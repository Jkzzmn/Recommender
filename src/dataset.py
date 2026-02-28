import torch
from torch.utils.data import Dataset

class MovieDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X) 
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MovieDatasetV2(Dataset):
    def __init__(self, df):
        # 1. ID 데이터 (Embedding layer에 들어갈 친구들)
        self.user_ids = torch.LongTensor(df['user_id'].values)
        self.item_ids = torch.LongTensor(df['movie_id'].values)
        
        # 2. 추가 피처 (나머지 모든 컬럼: 장르 19개 + 나이/연도 원핫 등)
        # user_id, movie_id, rating을 제외한 모든 컬럼을 추출
        age_cols = [c for c in df.columns if 'age_' in c]
        self.age_features = torch.FloatTensor(df[age_cols].values)
        genres_cols = [c for c in df.columns if c not in ['user_id', 'movie_id', 'rating']+ age_cols]
        self.genres_features = torch.FloatTensor(df[genres_cols].values)
        
        # 3. 정답 데이터
        self.targets = torch.FloatTensor(df['rating'].values)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # 모델의 forward(user_id, item_id, extra_feat) 형식에 맞춰 리턴
        return self.user_ids[idx], self.item_ids[idx], self.genres_features[idx], self.age_features[idx], self.targets[idx]