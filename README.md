# NLP-Okt

# [AI Hub 문화, 게임 콘텐트 분야 용어 말뭉치](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71614)
> 해당 파일에서 '레저'부분만 사용함

# 개요
* Okt를 사용하여 한국어 토크나이징. 명사, 명사+동사원형, 모든형태소로 각각 나눠서 학습하고 비교
* skipgram을 사용하여 중심 단어와 주변 단어를 학습하고 단어를 주면 주변 단어 후보를 반환
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
