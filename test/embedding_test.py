import warnings
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# FutureWarning 경고를 무시하도록 설정
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
# embedding_data = torch.load("../tools/qna/embedding_data.pt")
# df = pd.read_excel("D:/University/KSNU/3_1_머신러닝종합설계/haksa-bot/haksaBot/tools/qna/train_test.xlsx")
embedding_data = torch.load("../tools/qna/test_embedding_data.pt")
df = pd.read_excel("D:/University/KSNU/3_1_머신러닝종합설계/haksa-bot/haksaBot/tools/qna/test_train_data.xlsx")

sentence = "머신러닝 학점 몇이야?"
# sentence = "머신러닝 교수님 누구야?"
print("질문 문장 : ",sentence)
sentence = sentence.replace(" ","")
print("공백 제거 문장 : ", sentence)

sentence_encode = model.encode(sentence)
sentence_tensor = torch.tensor(sentence_encode)

cos_sim = util.cos_sim(sentence_tensor, embedding_data)
print(f"가장 높은 코사인 유사도 idx : {int(np.argmax(cos_sim))}")

best_sim_idx = int(np.argmax(cos_sim))
selected_qes = df['질문'][best_sim_idx]
print(f"선택된 질문 = {selected_qes}")

selected_qes_encode = model.encode(selected_qes)

score = np.dot(sentence_tensor, selected_qes_encode) / (np.linalg.norm(sentence_tensor) * np.linalg.norm(selected_qes_encode))
print(f"선택된 질문과의 유사도 = {score}")

answer = df['답변'][best_sim_idx]
print(f"\n답변 : {answer}\n")