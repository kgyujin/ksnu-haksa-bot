# 단어 사전 파일 생성 코드입니다.
# 챗봇에 사용하는 사전 파일

from utils.Preprocess import Preprocess
from tensorflow.keras import preprocessing
import pickle
import pandas as pd

# 말뭉치 데이터 읽어오기
time_expression = pd.read_csv('../../dataset/trans-dataset/시간표현탐지데이터.csv')
purpose = pd.read_csv('../../dataset/trans-dataset/용도별목적대화데이터.csv')
topic = pd.read_csv('../../dataset/trans-dataset/주제별일상대화데이터.csv')
common_sense = pd.read_csv('../../dataset/trans-dataset/일반상식.csv')
movie_review = pd.read_csv('../../dataset/trans-dataset/영화리뷰.csv')

time_expression.dropna(inplace=True)
purpose.dropna(inplace=True)
topic.dropna(inplace=True)
common_sense.dropna(inplace=True)
movie_review.dropna(inplace=True)

text1 = list(time_expression['text'])
text2 = list(purpose['text'])
text3 = list(topic['text'])
text4 = list(common_sense['query']) + list(common_sense['answer'])
text5 = list(movie_review['document'])

corpus_data = text1 + text2 + text3 + text4 + text5

# 말뭉치 데이터에서 키워드만 추출해서 사전 리스트 생성
# p = Preprocess()
p = Preprocess(word2index_dic='../../tools/dict/chatbot_dict.bin',
               userdic='../../utils/user_dic.tsv')
dict = []
for c in corpus_data:
    pos = p.pos(c)
    for k in pos:
        dict.append(k[0])

# 사전에 사용될 word2index 생성
# 사전의 첫 번째 인덱스에는 OOV 사용
tokenizer = preprocessing.text.Tokenizer(oov_token='OOV', num_words=100000)
tokenizer.fit_on_texts(dict)
word_index = tokenizer.word_index
print(len(word_index))

# 사전 파일 생성
f = open("chatbot_dict.bin", "wb")
try:
    pickle.dump(word_index, f)
except Exception as e:
    print(e)
finally:
    f.close()