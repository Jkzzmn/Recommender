import streamlit as st
import torch
import pandas as pd
import numpy as np
import yaml
from src.model import NeuralCF

# --- 1. 설정 및 리소스 로드 ---
@st.cache_resource
def load_resources():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 전처리된 최종 데이터 로드 (유저의 메타데이터를 참조하기 위함)
    data = pd.read_csv(config['path']['data_v2_path'])
    
    # 모델 파라미터 설정
    n_users = data['user_id'].max() + 1
    n_movies = data['movie_id'].max() + 1
    
    # extra_cols 정의 (학습 때와 동일한 순서여야 함)
    target_col = 'rating'
    id_cols = ['user_id', 'movie_id']
    extra_cols = [c for c in data.columns if c not in id_cols + [target_col]]
    extra_dim = len(extra_cols)
    
    # 모델 로드
    model = NeuralCF(
        n_users, 
        n_movies, 
        embedding_dim=config['model_v2']['embedding_dim'],
        extra_dim=extra_dim
    )
    model.load_state_dict(torch.load(config['path']['model_v2_path'], map_location='cpu'))
    model.eval()
    
    # 영화 제목 매핑 (u.item 활용 또는 데이터프레임 내 movie_title이 있다면 사용)
    # 여기서는 편의상 movie_id로 표시하거나 별도의 영화 정보 파일을 merge해서 사용하세요.
    return data, model, extra_cols, config

data, model, extra_cols, config = load_resources()

# --- 2. 웹 화면 구성 ---
st.set_page_config(page_title="대현의 AI 추천 v2", page_icon="🍿")
st.title("🎬 대현의 지능형 영화 추천 시스템 (v2)")
st.markdown(f"**현재 반영된 피처:** 장르, 나이대, 성별, 직업 (총 {len(extra_cols)}개)")

user_id_input = st.number_input("유저 ID를 입력하세요", min_value=0, step=1)

if st.button("추천 받기"):
    if user_id_input in data['user_id'].unique():
        # 1. 해당 유저의 메타데이터(나이, 성별 등) 가져오기
        user_meta = data[data['user_id'] == user_id_input].iloc[0]
        
        # 2. 모든 영화 리스트 준비
        all_movie_ids = data['movie_id'].unique()
        n_all_movies = len(all_movie_ids)
        
        # 3. 모델 입력을 위한 텐서 생성
        u_tensor = torch.LongTensor([user_id_input] * n_all_movies)
        m_tensor = torch.LongTensor(all_movie_ids)
        
        # 유저 메타데이터(extra_features) 복제
        # 해당 유저의 고정된 피처(장르 제외한 유저 피처 + 영화별 장르를 합쳐야 정확하지만, 
        # 일단 가장 간단하게 해당 유저가 본 평균적인 특징으로 예측하거나 
        # 영화별 장르 정보를 매핑해서 가져와야 합니다.)
        
        # [정석적인 방법] 각 영화의 장르 정보와 유저의 정보를 합친 extra_features 행렬 생성
        movie_info = data.drop_duplicates('movie_id').set_index('movie_id')[extra_cols]
        user_extra_features = movie_info.copy()
        
        # 유저 고유 정보(나이, 성별 등)는 동일하게 복사하고 장르만 영화 정보를 따름
        user_cols = [c for c in extra_cols if 'gen_' in c or 'occ_' in c or 'age_' in c]
        for col in user_cols:
            user_extra_features[col] = user_meta[col]
            
        e_tensor = torch.FloatTensor(user_extra_features.loc[all_movie_ids].values)
        
        # 4. 예측
        with torch.no_grad():
            predictions = model(u_tensor, m_tensor, e_tensor)
        
        # 5. 결과 정리
        top_k = 10
        scores, indices = torch.topk(predictions, k=top_k)
        
        st.subheader(f"✅ 유저 {user_id_input}님께 추천하는 영화")
        
        rec_data = []
        for i in range(top_k):
            m_id = all_movie_ids[indices[i].item()]
            score = scores[i].item()
            rec_data.append({"순위": i+1, "Movie ID": m_id, "추천 점수": f"{score:.2f}점"})
            
        st.table(pd.DataFrame(rec_data))
        st.success("유저님의 나이대와 직업, 영화 장르 취향을 분석하여 최적의 영화를 찾아냈습니다!")
    else:
        st.error("데이터셋에 없는 유저입니다.")