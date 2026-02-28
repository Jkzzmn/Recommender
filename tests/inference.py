import torch
import joblib
import pandas as pd
from src.model import MatrixFactorization

# --- Step 1. 필요한 설정 및 데이터 로드 ---
# 1. 저장해둔 번역기(딕셔너리) 불러오기
user2idx = joblib.load('models/user2idx.joblib')
movie2idx = joblib.load('models/movie2idx.joblib')
# 반대로 인덱스를 넣으면 ID를 주는 주소록도 만듭니다.
idx2movie = {v: k for k, v in movie2idx.items()}

# 2. 영화 제목 정보를 가져오기 위해 원본 데이터 로드
# (실제 서비스라면 DB에서 가져오겠지만, 여기선 processed_data.csv를 활용할게요)
data = pd.read_csv('data/processed_data.csv')
movie_titles = data[['movie_id', 'movie_title']].drop_duplicates().set_index('movie_id')

# 3. 모델 설정 및 가중치 불러오기
n_users = len(user2idx)
n_movies = len(movie2idx)
model = MatrixFactorization(n_users, n_movies, embedding_dim=20)

# 학습된 파라미터(.pth)를 모델에 덮어씌웁니다.
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval() # 평가 모드 (중요!)

# --- Step 2. 추천 함수 만들기 ---
def recommend_movies(user_id, top_n=10):
    # 1. 실제 유저 ID를 모델용 인덱스로 변환
    if user_id not in user2idx:
        return "존재하지 않는 유저입니다."
    
    u_idx = user2idx[user_id]
    
    # 2. 모든 영화에 대해 점수 예측하기
    # 유저 인덱스는 고정하고, 모든 영화 인덱스(0 ~ n_movies-1)를 준비합니다.
    all_movie_indices = torch.arange(n_movies)
    user_indices = torch.tensor([u_idx] * n_movies)
    
    with torch.no_grad(): # 예측 시에는 기울기 계산 안 함 (속도 향상)
        predictions = model(user_indices, all_movie_indices)
    
    # 3. 예측 점수가 높은 순으로 정렬
    # topk 함수는 상위 n개의 값과 인덱스를 반환합니다.
    top_scores, top_indices = torch.topk(predictions, k=top_n)
    
    # 4. 결과 출력용 제목 매핑
    results = []
    for i in range(top_n):
        idx = top_indices[i].item()
        score = top_scores[i].item()
        movie_real_id = idx2movie[idx]
        title = movie_titles.loc[movie_real_id, 'movie_title']
        results.append((title, score))
        
    return results

# --- Step 3. 실제 실행 ---
target_user = 196 # 테스트해보고 싶은 유저 ID를 넣어보세요.
print(f"\n[유저 {target_user}님을 위한 추천 리스트]")
recommendations = recommend_movies(target_user)

for i, (title, score) in enumerate(recommendations):
    print(f"{i+1}. {title} (예상 평점: {score:.2f})")