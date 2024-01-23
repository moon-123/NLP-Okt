# NLP-Okt

# AI Hub [문화, 게임 콘텐트 분야 용어 말뭉치]
> 해당 파일에서 '레저'부분만 사용함 
[데이터 출처](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71614)


# 개요
* Okt를 사용하여 한국어 토크나이징. 명사, 명사+동사원형, 모든형태소로 각각 나눠서 학습하고 비교
* skipgram을 사용하여 중심 단어와 주변 단어를 학습하고 단어를 주면 주변 단어 후보를 반환

# 토크나이징
* 한국어 형태소는 토큰화하기 매우 까다롭다.
* 중심 단어와 주변 단어의 관계를 파악하는 것이 목적이다.
* 문장에 사용된 어절 그대로 자를 것인지, 형태소별로 자를 것인지 선택을 해야했다.
* 명사만, 명사+동사원형 각각을 테스트해보고 결과를 비교할 것이다.
--------
# 1. 데이터 불러오기

* 파이썬 json 라이브러리를 사용하여 JSON파일 로드
  
```python
import json

with open('/content/drive/MyDrive/KDT/자연어 처리/data/용례_레저.json', 'r') as file:
    json_data = json.load(file)
```

* json 데이터 형식
```
{'id': 3087876,
 'sentence': '김치볶음 해놓은거에 밥비벼먹고 카레 해먹고',
 'tokens': [{'start': 0,
   'length': 4,
   'sub': '김치볶음',
   'facet': '구체물',
   'term_id': 134217,
   'sense_no': 1}],
 'sense_no': 1,
 'source': {'uri': 'https://bbs.ruliweb.com/community/board/300143/read/58933395',
  'text': '일주일동안 탄수화물 위주로만 먹었어\r\n\r\n\r\n김치볶음 해놓은거에 밥비벼먹고 카레 해먹고\r\n\r\n\r\n점심에는 김밥에 쫄면에 돈까스 먹었는데\r\n\r\n\r\n그래도 단백질이 먹고싶다',
  'written_at': '2022-10-15T08:04:00'}}
```
text에서 추출한 sentence항목을 train데이터로 사용.

# 2. Okt(명사만 추출)

```shell
pip install KoNLPy
```

```python
from konlpy.tag import Okt
okt = Okt()
```

### 2-1. 명사 토크나이징
* Okt의 nouns()를 사용하여 명사만 추출한다.
  
```python
tokenized_sentence_list = []

for sentence in sentence_list:
    tokenized_sentence = okt.nouns(sentence)
    tokenized_sentence_list.append(tokenized_sentence)
```

### 2-2. 불용어 제거
* 불용어와 특수문자를 제거하는 전처리를 진행한다.

```python
added_sw = ['거', '점', '걸', '더', '게', '것', '데', '참', '뭘', '쭉']
stopwords_list = set(open('/content/drive/MyDrive/KDT/자연어 처리/data/stopword.txt').read().split('\n')).union(added_sw)
len(stopwords_list)
pre_sentence_list = []
for sentence in tokenized_sentence_list:
    word_list = []
    for word in sentence:
        if word not in stopwords_list:
            word_list.append(word)

    pre_sentence_list.append(word_list)
```

* 빈 리스트 제거

```python
sentence_list = []
for ps in pre_sentence_list:
    if len(ps) != 0:
        sentence_list.append(ps)

print(len(sentence_list))
```

### 2-3. Tokenizer로 단어 라벨링

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentence_list)

# key가 word
word2idx = tokenizer.word_index

# key가 idx
idx2word = {value : key for key, value in word2idx.items()}

vocab_size = len(idx2word)

# 단어 시퀀스를 숫자 시퀀스로
encoded = tokenizer.texts_to_sequences(sentence_list)
```

### 2-4. skip-gram

```python
from tensorflow.keras.preprocessing.sequence import skipgrams

skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=5) for sample in encoded[:500]]
# negative_samples=1, shuffle=True
```
* skipgrams의 경우 중심 단어의 주변 단어가 label이 되는 방식이다.
* 어떤 중심단어와 주변단어 후보의 쌍이 입력되었을 때 후보가 실제 주변단어가 맞다면 1로 라벨링을 한다
* 이 기법은 정확하게는 skip-gram Negative Sampling(SGNS)이라고 한다.
* negative_samples가 1이면 입력 시퀀스의 각 요소마다 부정적인 쌍이 하나씩 생성된다.

### 2-5. 학습을 위한 모델 만들기

* 모델은 기존에 예제용으로 사용하던 모델을 가져옴.
  
```python
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Reshape, Activation, Input, Dot
from tensorflow.keras.utils import plot_model

embedding_dim = 100

w_inputs = Input(shape=(1,), dtype='int32')
word_embedding = Embedding(vocab_size, embedding_dim)(w_inputs)

c_inputs = Input(shape=(1,), dtype='int32')
context_embedding = Embedding(vocab_size, embedding_dim)(c_inputs)

dot_product = Dot(axes=2)([word_embedding, context_embedding])
dot_product = Reshape((1,), input_shape=(1, 1))(dot_product)
output = Activation('sigmoid')(dot_product)

model = Model(inputs=[w_inputs, c_inputs], outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam')
```

### 2-6. 학습

```python
import numpy as np

for epoch in range(100):
    loss = 0
    for _, elem in enumerate(skip_grams):
        if elem[0]:
            first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
            second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
            labels = np.array(elem[1], dtype='int32')
            X = [first_elem, second_elem]
            Y = labels
            loss += model.train_on_batch(X, Y)
    print('Epoch: ', epoch+1, 'Loss: ', loss)
```
----------
# 3. Okt(명사+동사원형)


* 입력 시퀀스를 500개, epoch을 100으로 주니 시간이 오래 걸리고 50번째 epoch부터는 수렴 속도가 굉장히 느려졌다.
* 주어진 시간이 많지 않아 이대로 실험을 종료했지만 학습 조기 종료를 적당히 주는것이 좋을 것 같다.
  
