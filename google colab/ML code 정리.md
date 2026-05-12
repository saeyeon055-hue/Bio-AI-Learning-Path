# SVM
=> 대상: 환자의 혈압, 혈당, 나이를 보고 '당뇨 여부'를 판단하는 것처럼 데이터가 하나하나 독립적일 때 사용
=> 변환: 표(Table) 형태의 데이터를 숫자로 바꾸는 과정이 위주
=> 평가: '100명 중 몇 명 맞췄나?'하는 정확도(Accuracy) 중요


## 1. open
```
with open(file_path, 'r', encoding='utf8') as inFile:
```
- file_path(어떤 파일을?): 열고자 하는 파일의 이름이나 경로. 컴퓨터에게 'data.txt 가져와'라고 알려주는 주소 역할
- 'r'(어떻게?): Read의 약자. '안에 있는 내용을 읽기만 할 거야' 라는 뜻. 내용을 쓰고 싶을때는 'w'를 사용하는데, 실수로 파일을 지우거나 수정하는 걸 방지하기 위해 읽을 때는 꼭 'r'을 사용함.
- encoding='utf8'(어떤 언어로?): 컴퓨터는 0과 1만 알기 때문에, 글자를 숫자로 바꾸는 규칙 필요. UTF-8은 한글을 포함해 전 세계 모든 문자를 깨짐 없이 읽을 수 있는 가장 표준적인 규칙(규격).
- with(안전 장치): 파일을 열었으면 작업 후 반드시 닫아야(close) 컴퓨터 메모리가 낭비되지 않는다. with를 사용하면, 들여쓰기 된 안쪽 코드들이 다 실행된 후 파이썬이 알아서 파일을 안전하게 닫아준다.

```
  lines=inFile.readlines()
```
- inFile: open 함수를 통해 열린 파일 객체 그 자체를 부르는 별명
- .readlines(): 이 상자 안에 있는 내용을 한 줄씩 다 읽어서 리스트(list)형태로 꺼내는 명령어
- lines = ...: 그렇게 꺼내진 데이터들을 lines라는 이름의 바구니에 담아두고, 앞으로 코드에서 사용하겠다는 의미

## 2. 데이터 읽기
```
x_data, y_data=[],[]
for line in lines:
  line = line.strip().split('\t')
  sentence, label = line[1], line[0]
  x_data.append(sentence)
  y_data.append(label)

print('x_data의 개수 : ' + str(len(x_data)))
print('y_data의 개수 : ' + str(len(y_data)))
```
- line.strip(): 양쪽 끝에 붙은 불필요한 공백이나 줄바꿈 문자(\n)을 제거하는 기능.
- split('\t'): 데이터가 '탭' 키로 구분되어 있다는 뜻. 엑셀 데이터를 텍스트로 저장하면 보통 칸 사이가 탭으로 구분되는데, 이를 기준으로 나눈다는 의미
- label: 데이터의 정답(category)을 의미. 스팸 메일 분류라면 0(정상), 1(스팸)이 될 수 있고, 약물 분류라면 0(항생제), 1(소화제), 2(해열제)처럼 여러 개가 될 수도 있다.
- append: 리스트(list)라는 주머니에 넣는 것. 비어있는 리스트에 새로운 데이터 sentence, label을 하나씩 뒤로 이어 붙이는 작업.
- str(len(x_data)): len()은 숫자를 return하는데, '문자'와'숫자'를 더하기 할 수 없기 때문에 숫자를 다시 문자열(str)로 바꾸어 출력

## 3. 데이터 변환 (자질 설계)
```
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer()
```
- 텍스트 데이터를 SVM 모델이 이해할 수 있는 '숫자'로 바꾸기 위한 준비 단계.
- TensorFlow라는 라이브러리 안에 있는 Tokenizer라는 도구를 가져오는 과정

### spam, ham 라벨을 대응하는 index로 치환하기위한 딕셔너리
```
label2index_dict={'spam':0, 'ham':1}
```
- 모델이 계산할 수 있도록 정답을 숫자로 매핑하는 기준표. 모델이 '0'이라 답했을 때, 스팸이구나 라고 우리가 알아보기 이해 미리 규칙을 정해두는 것.

### X_data를 사용하여 딕셔너리 생성
```
tokenizer.fit_on_texts(x_data)
```
- fit은 SVM 모델을 학습시키는 것이 아니라, 단어 사전을 만드는 과정.
- 두 가지 종류의 fit이 머신러닝 라이브러리에서 사용된다.
- 1) 모델 학습을 위한 fit(Decision Tree, SVM 등): 데이터(X)와 정답(Y) 사이의 규칙을 찾아내는 역할로, 학생이 시험 공부를 위해서 문제 풀이 능력을 갖추는 것.
  2) 전처리를 위한 fit(Tokenizer Scaler 등): 데이터(X)에 어떤 단어들이 있는지 훑어보고 기준(번호표/사전)을 만드는 것으로, 공부를 시작하기 전, 교과서 뒤에 있는 인덱스를 정리하는 것,

  
### 데이터 변환/ 자질 설계 핵심
```
indexing_x_data=tokenizer.texts_to_sequences(x_data)
```
- texts_to_sequences: 실제 문장을 위에서 만든 번호표(index)의 나열로 바꾸어, SVM이 수학적으로 계산을 할 수 있는 상태로 만드는 작업을 한다.

## 4. SVM 학습
```
from sklearn.svm import SVC

# 전체 데이터를 9:1의 비율로 나누어 학습 및 평가 데이터로 사용
number_of_train=int(len(indexing_x_data)*0.9)

train_x=indexing_x_data[:number_of_train]
train_y=indexing_y_data[:number_of_train]
test_x=indexing_x_data[number_of_train:]
test_y=indexing_y_data[number_of_train:]

print('train_x의 개수 : ' + str(len(train_x)))
print('train_y의 개수 : ' + str(len(train_y)))
print('test_x의 개수 : ' + str(len(test_x)))
print('test_y의 개수 : ' + str(len(test_y)))

svm=SVC(kernel='linear', C=1e10)
svm.fit(train_x,train_y)
```
- kernel='linear'(분류 기준의 모양): 데이터들 사이에 '직선'으로 칼같이 경계선을 긋겠다는 뜻(성분 수치가 일정 선을 넘으면 합격, 아니면 불합격). 데이터가 복잡하면 곡선(rbf)를 쓰기도 하지만, 텍스트 데이터는 보통 직선으로도 충분하다.
- C=1e10(규제/엄격함의 정도): 오답을 얼마나 허용할지 결정하는 값. 1e10은 아주 큰 숫자로, 매우 엄격하게 모든 데이터를 완벽하게 분류하는 선을 찾으라는 명령(값이 작을수록 '몇 개 틀려도 좋으니 선을 좀 부드럽게 그려'라는 뜻)
- svm.fit(train_x,train_y): 모델 훈련시키는 명령어

## 5. SVM 평가
```
# 1) 수치로 보는 성적표(Accuracy)
predict =svm.predict(test_x)

correct_count=0
for index in range(len(predict)):
  if(test_y[index] == predict[index]):
    correct_count+=1

accuracy = 100.0*correct_count/len(test_y)

print('Accuracy: ' + str(accuracy))
```
- 1)과 2) 두 부분으로 나뉘는 이유는 '어떤 문장에서 틀렸는지'를 알아야 모델을 개선할 수 있기 때문

```
# 2) 사람이 눈으로 직접 확인하는 상세 결과(Detail)
index2label = {0:'spam', 1:'ham'}

test_x_word = tokenizer.sequences_to_texts(test_x)

for index in range(len(test_x_word)):
  print()
  print('문장 : ', test_x_word[index])
  print('정답 : ', index2label[test_y[index]])
  print('모델 출력 : ', index2label[predict[index]])
```
- index2label은 index2label_dict와 반대 버전. 숫자로 된 결과(0,1)를 다시 글자('spam','ham')로 바꾼다.
- tokenizer.sequences_to_texts(test_x): 숫자로 변환되어 있던 text_x를 원래의 문장으로 복구한다.


# CRF
=> 대상: 단백질 서열, 약물 복용 순서, 문장처럼 데이터 앞뒤의 순서가 중요할 때 사용
=> 변환: 현재 단어뿐만 아니라 '앞 단어는 뭐였지?', '뒤 단어는 뭐지?'같은 주변 맥락을 추출해서 sent2feature 처럼 복잡한 딕셔너리 만든다.
=> 평가: 하나하나 맞추는 것보다, 전체 시퀀스를 얼마나 완벽하게 맞췄는지, 혹은 특정 개체를 얼마나 잘 찾아냈는지(F1-score) 중요

## 2. 데이터 읽기
```
datas = []
for line in lines:
  pieces = line.strip().split('\t')
  eumjeol_sequence, label = pieces[0].split(), pieces[1].split() # pieces[0]: 음절 열, pieces[1]: 레이블
  datas.append((eumjeol_sequence, label))
```
- .split()의 추가 사용: pieces[0]과 pieces[1] 뒤에 .split()이 한 번 더 붙었다. 이는 공백(space)을 기준으로 문자열을 쪼개서 리스트로 만들라는 뜻.
- 예) .split() 추가 사용 없을 때: eumjeol_sequence: [나 는 사 과 가 좋 아], label: [ NP SBJ OBJ ...]
-     .split() 추가 사용 할 때: eumjeol_sequence: ['나', '는', '사', '과', '가', '좋', '아'], label: ['NP', 'SBJ', 'OBJ', ...]

## 3. 데이터 변환 (자질 설계)
```
import sklearn_crfsuite
from sklearn_crfsuite import metrics
```

```
def sent2feature(eumjeol_sequence):
  features = []
  sequence_length = len(eumjeol_sequence)
  for index, eumjeol in enumerate(eumjeol_sequence):
```
- def sent2feature(eumjeol_sequence): sent2feature라는 이름의 함수 생성.
- sequence_length = len(eumjeol_sequence): 문장의 총 글자 수 확인. 문장의 길이를 알아야 '현재 글자가 마지막 글자인지(EOS)', 혹은 '첫 글자인지(BOS)'를 판단할 수 있기 때문.
- for index,eumjeol in enumerate(eumjeol_sequence): enumerate는 리스트에서 글자(eumjeol)만 꺼내는 게 아니라, 그 글자가 몇 번째인지 번호(index)까지 세어주는 명령어
- 예) ['나', '는', '사', '과']라는 리스트가 있다면: 1회전: index=0, eumjeol='나', 2회전: index=1, eumjeol='는' ...

## 4. 데이터 생성
```
train_x, train_y = [], []
for eumjeol_sequence, label in train_datas:
  train_x.append(sent2feature(eumjeol_sequence))
  train_y.append(label)

test_x, test_y = [], []
for eumjeol_sequence, label in test_datas:
  test_x.append(sent2feature(eumjeol_sequence))
  test_y.append(label)
```
- train_x.append(sent2feature(eumjeol_sequence))
  test_x.append(sent2feature(eumjeol_sequence)) 여기서 sent2feature가 왜 들어가는가?
  텍스트를 '특징 세트(사전)'로 변환하기 위해서.
  예) 변환 전 ['나', '는', '사', '과'] (기계는 이게 뭔지 모름)
      변환 후(sent2feature 실행 시) '나': {첫글자: True, 글자: '나', 다음글자: '는'}
                                   '는': {첫글자: False, 글자: '는', 이전글자: '나', 다음글자: '사'}

# CRF 라이브러리는 단순한 글자가 아니라, 딕셔너리 형태의 특징 세트를 입력받도록 설계되어 있음. 그래서 append 하기 전에 반드시 sent2feature 공장을 거쳐서 가공된 데이터를 넣어줘야 함.

## 5. CRF 학습
```
crf = sklearn_crfsuite.CRF()
crf.fit(train_x, train_y)
```

## 6. CRF 평가
```
predict = crf.predict(test_x)
```

```
# 1) 수치로 보는 성적표(Accuracy)
print('Accuracy score : ' + str(metrics.flat_accuracy_score(test_y, predict)))
print()

# 2) 사람이 눈으로 직접 확인하는 상세 결과(Detail)
print('10개의 데이터에 대한 모델 출력과 실제 정답 비교')
print()


def show_predict_result(test_datas, predict):
  for inex_1 in range(len(test_datas)):
    eumjeol_sequence, correct_labels = test_datas[inex_1]
    predict_labels = predict[inex_1]

    correct_sentence, predict_sentence = '', ''
    for index_2 in range(len(eumjeol_sequence)):
      if (index_2== 0):
        correct_sentence += eumjeol_sequence[index_2]
        predict_sentence += eumjeol_sequence[index_2]
        continue

      if (correct_labels[index_2] == 'B'):
        correct_sentence += ' '
      correct_sentence += eumjeol_sequence[index_2]

      if (predict_labels[index_2] == 'B'):
        predict_sentence += ' '
      predict_sentence += eumjeol_sequence[index_2]

    print('정답 문장 : ' + correct_sentence)
    print('예측 문장 : ' + predict_sentence)
    print()

show_predict_result(test_datas[:10], predict[:10])
```
