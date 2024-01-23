# NLP-Okt

# AI Hub 문화, 게임 콘텐트 분야 용어 말뭉치
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

# 2. Okt

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
stopwords_list = set(open('/content/drive/MyDrive/KDT/자연어 처리/data/stopword.txt').read().split('\n'))
len(stopwords_list)
pre_sentence_list = []
for sentence in tokenized_sentence_list:
    word_list = []
    for word in sentence:
        if word not in stopwords_list:
            word_list.append(word)

    pre_sentence_list.append(word_list)

len(pre_sentence_list)
```
