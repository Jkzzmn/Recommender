import streamlit as st
import torch
import joblib
import pandas as pd
from src.model import MatrixFactorization

# --- 1. 데이터 및 모델 로드 (캐싱을 통해 속도 향상) ---
@st.cache_resource
def load_resources():
    # 번역기 로드
    user2idx = joblib.load('models/user2idx.joblib')
    movie2idx = joblib.load('models/movie2idx.joblib')
    idx2movie = {v: k for k, v in movie2idx.items()}
    
    # 영화 정보 로드
    data = pd.read_csv('data/processed_data.csv')
    movie_titles = data[['movie_id', 'movie_title']].drop_duplicates().set_index('movie_id')
    
    # 모델 로드
    n_users = len(user2idx)
    n_movies = len(movie2idx)
    model = MatrixFactorization(n_users, n_movies, embedding_dim=20)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    return user2idx, movie2idx, idx2movie, movie_titles, model

user2idx, movie2idx, idx2movie, movie_titles, model = load_resources()

# --- 2. 웹 화면 구성 ---
st.title("🎬 대현의 AI 영화 추천 시스템")
st.write("유저 ID를 입력하면 취향에 맞는 영화를 추천해 드립니다.")

# 사용자 입력창
user_id_input = st.number_input("유저 ID를 입력하세요 (예: 1, 196, 55)", min_value=1, step=1)

if st.button("추천 받기"):
    if user_id_input in user2idx:
        u_idx = user2idx[user_id_input]
        
        # 모든 영화에 대해 점수 예측
        all_movie_indices = torch.arange(len(movie2idx))
        user_indices = torch.tensor([u_idx] * len(movie2idx))
        
        with torch.no_grad():
            predictions = model(user_indices, all_movie_indices)
        
        # 상위 10개 추출
        top_scores, top_indices = torch.topk(predictions, k=10)
        
        st.subheader(f"🍿 유저 {user_id_input}님을 위한 TOP 10 추천")
        
        # 결과 출력 (표 형식)
        rec_list = []
        for i in range(10):
            idx = top_indices[i].item()
            score = top_scores[i].item()
            movie_real_id = idx2movie[idx]
            title = movie_titles.loc[movie_real_id, 'movie_title']
            rec_list.append({"순위": i+1, "영화 제목": title, "예측 평점": f"{score:.2f}점"})
        
        st.table(pd.DataFrame(rec_list))
    else:
        st.error("데이터셋에 존재하지 않는 유저 ID입니다.")