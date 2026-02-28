import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model import MatrixFactorization
from src.dataset import MovieDataset

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data = pd.read_csv(config['path']['data_path'])

n_users = data['user'].nunique()
n_movies = data['movie'].nunique()

X = data[['user', 'movie']].values
y = data['rating'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = MovieDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)

model = MatrixFactorization(n_users, n_movies, embedding_dim=config['model']['embedding_dim'])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

epochs = config['train']['epochs']
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, ratings in train_loader:

        users = inputs[:, 0]
        movies = inputs[:, 1]

        preds = model(users, movies)
        loss = criterion(preds, ratings.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# --- Step 6. 모델 저장 (중요!) ---
# 학습이 끝난 모델의 '무게추(Weight)'를 저장합니다. 그래야 나중에 웹에서 바로 쓰죠.
torch.save(model.state_dict(), config['path']['model_path'])
print("학습 완료 및 모델 저장됨!")