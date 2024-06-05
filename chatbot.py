import json
import pandas as pd
import tensorflow as tf
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

from utils.Preprocess import Preprocess
from utils.FindAnswer import FindAnswer
from model.IntentModel import IntentModel
from tools.qna.create_embedding_data import create_embedding_data

# tensorflow gpu 메모리 할당
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=2048)])
    except RuntimeError as e:
        print(e)

# 로그 기능 구현
from logging import handlers
import logging

# log settings
LogFormatter = logging.Formatter('%(asctime)s,%(message)s')

# handler settings
LogHandler = handlers.TimedRotatingFileHandler(filename='./logs/chatbot.log', when='midnight', interval=1,
                                               encoding='utf-8')
LogHandler.setFormatter(LogFormatter)
LogHandler.suffix = "%Y%m%d"

# logger set
Logger = logging.getLogger()
Logger.setLevel(logging.ERROR)
Logger.addHandler(LogHandler)

# 전처리 객체 생성
try:
    p = Preprocess(word2index_dic='./tools/dict/chatbot_dict.bin',
                   userdic='./utils/user_dic.tsv')
    print("텍스트 전처리기 로드 완료..")
except Exception as e:
    print(f"텍스트 전처리기 로드 실패: {e}")

# 의도 파악 모델
try:
    intent = IntentModel(model_name='./model/intent_model.h5', preprocess=p)
    print("의도 파악 모델 로드 완료..")
    print(intent)  # 추가: intent 객체 확인
except Exception as e:
    print(f"의도 파악 모델 로드 실패: {e}")


# 엑셀 파일 로드
try:
    df = pd.read_excel('tools/qna/train_test.xlsx')
    for column in df.columns:
        df[column] = df[column].astype(str)
    print(f"엑셀 파일 컬럼: {df.columns}")  # 추가: 엑셀 파일 컬럼 확인
    print("엑셀 파일 로드 완료..")
except Exception as e:
    print(f"엑셀 파일 로드 실패: {e}")

# pt 파일 갱신 및 불러오기
try:
    create_embedding_data = create_embedding_data(df=df, preprocess=p)
    create_embedding_data.create_pt_file()
    embedding_data = torch.load('tools/qna/embedding_data.pt')
    print("임베딩 pt 파일 갱신 및 로드 완료..")
except Exception as e:
    print(f"임베딩 pt 파일 갱신 및 로드 실패: {e}")


def process_query(query):
    try:
        # print(f"사용자 질문: {query}")  # 디버깅 메시지 추가
        # 의도 파악
        intent_pred = intent.predict_class(query)
        # print(f"예측된 의도 인덱스: {intent_pred}")  # 디버깅 메시지 추가
        intent_name = intent.labels[intent_pred]
        # print(f"예측된 의도 이름: {intent_name}")  # 디버깅 메시지 추가

        # 답변 검색
        f = FindAnswer(preprocess=p, df=df, embedding_data=embedding_data)
        selected_qes, score, answer, query_intent = f.search(query, intent_name)

        # if score < 0.6:
        #     answer = "부정확한 질문이거나 답변할 수 없습니다.\n 죄송합니다 :("
        #     Logger.error(f"{query},{intent_name},{selected_qes},{query_intent},{score}")

        response = {
            "Query": selected_qes,
            "Answer": answer,
            "Intent": intent_name
        }
        return response

    except Exception as ex:
        print(f"오류 발생: {ex}")
        Logger.error(f"오류 발생: {ex}")
        return None


if __name__ == '__main__':
    print("챗봇이 시작되었습니다. 질문을 입력하세요 ('exit'를 입력하면 종료됩니다).")

    while True:
        user_query = input("질문: ")
        if user_query.lower() == 'exit':
            print("챗봇을 종료합니다.")
            break

        response = process_query(user_query)
        if response:
            print(f"답변: {response['Answer']}")
