import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import torch
from sentence_transformers import SentenceTransformer

# 파일 경로와 모델 로드
train_file = "D:/University/KSNU/3_1_머신러닝종합설계/haksa-bot/haksaBot/tools/qna/train_test.xlsx"
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 엑셀 파일 읽기
df = pd.read_excel(train_file)

# NaN 값을 빈 문자열로 대체
df['질문'] = df['질문'].fillna('')

# 임베딩 벡터 생성
df['embedding_vector'] = df['질문'].progress_map(lambda x: model.encode(x))

# 임베딩 벡터를 포함한 엑셀 파일 저장
df.to_excel("train_data_embedding.xlsx", index=False)

# 임베딩 벡터를 텐서로 변환하여 저장
embedding_data = torch.tensor(df['embedding_vector'].tolist())
torch.save(embedding_data, 'embedding_data.pt')
print("임베딩 pt 파일 생성 완료..")