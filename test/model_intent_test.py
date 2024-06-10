# test/model_intent_test.py

from tensorflow.keras.models import load_model
from utils.Preprocess import Preprocess
from model.IntentModel import IntentModel
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf

# Register the custom activation function
get_custom_objects().update({"softmax_v2": tf.nn.softmax})

p = Preprocess(word2index_dic='../tools/dict/chatbot_dict.bin',
               userdic='../utils/user_dic.tsv')

intent = IntentModel(model_name='../model/intent_model.h5', preprocess=p)

query = "이종찬 교수님 연구실 전화번호 알려줘."
predict = intent.predict_class(query)
predict_label = intent.labels[predict]
print("="*30)
print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)


query = "2학기 개강은 언제야?"
predict = intent.predict_class(query)
predict_label = intent.labels[predict]
print("="*30)
print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)


query = "1학년 전공 뭐 있어?"
predict = intent.predict_class(query)
predict_label = intent.labels[predict]
print("="*30)
print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)


query = "머신러닝 강의 교수님 누구야?"
predict = intent.predict_class(query)
predict_label = intent.labels[predict]
print("="*30)
print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)


query = "머신러닝 강의실 어디야?"
predict = intent.predict_class(query)
predict_label = intent.labels[predict]
print("="*30)
print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)

query = "2학기 종강 언제야?"
predict = intent.predict_class(query)
predict_label = intent.labels[predict]
print("="*30)
print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)
print("="*30)