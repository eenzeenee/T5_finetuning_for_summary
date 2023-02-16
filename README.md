# T5_finetuning_for_summary

한국어 데이터셋을 활용하여 T5 모델을 text summarization task를 위해 finetuning 하는 코드입니다.

## Repository 구조

```
├── README.md
├── requirements.txt
├── t5_lib.txt
├── train.py
├── utils.py
├── dataloader.py
└── infer.py
```

## Requirements
- Python 3 (tested on 3.8)
- CUDA (tested on 11.3)

`conda create -n t5 python=3.8`  
`conda install -c anaconda numpy`  
`conda install -c conda-forge transformers`  
`conda install -c conda-forge datasets`  
`conda install -c anaconda nltk`

or

`conda env create -n t5 python=3.8 -f requirements.txt`

and then,
`conda activate t5`

## Training

`python train.py`

If you want to change hyper-parameters for training,  
`python train.py --num_train_epochs 5 --train_batch_size 16 ...`

## Inference

`python infer.py`

If you want to change hyper-parameters for inference,

`python train.py --file_path ./data/test.json ...`

## Data format

In **train/val/test.json** file,

```
[{'source':'...', 'target':'...'}, {'source':'...', 'target':'...'}, ...]
```

After inference, in **result.json** file,

```
[{'source':'...', 'target':'...'}, {'source':'...', 'target':'...'}, ...]
```


# Evalutation Result

- [Korean Paper Summarization Dataset(논문자료 요약)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=90)
  ```
  ROUGE-2-R 0.09868624890432466
  ROUGE-2-P 0.9666714545849712
  ROUGE-2-F 0.17250881441169427
  ```
- [Korean Book Summarization Dataset(도서자료 요약)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=93)
  ```
  ROUGE-2-R 0.1575686156943213
  ROUGE-2-P 0.9718318136896944
  ROUGE-2-F 0.26548116834852586
  ```
- [Korean Summary statement and Report Generation Dataset(요약문 및 레포트 생성 데이터)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=90)
  ```
  ROUGE-2-R 0.0987891733555808
  ROUGE-2-P 0.9276946867981899
  ROUGE-2-F 0.17726493110448185
  ```

## Finetuned model

### model link

[huggingface link](https://huggingface.co/eenzeenee/t5-base-korean-summarization)

### usage
```
import nltk
nltk.download('punkt')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained('eenzeenee/t5-base-korean-summarization')
tokenizer = AutoTokenizer.from_pretrained('eenzeenee/t5-base-korean-summarization')

prefix = "summarize: "
sample = """
    안녕하세요? 우리 (2학년)/(이 학년) 친구들 우리 친구들 학교에 가서 진짜 (2학년)/(이 학년) 이 되고 싶었는데 학교에 못 가고 있어서 답답하죠? 
    그래도 우리 친구들의 안전과 건강이 최우선이니까요 오늘부터 선생님이랑 매일 매일 국어 여행을 떠나보도록 해요. 
    어/ 시간이 벌써 이렇게 됐나요? 늦었어요. 늦었어요. 빨리 국어 여행을 떠나야 돼요. 
    그런데 어/ 국어여행을 떠나기 전에 우리가 준비물을 챙겨야 되겠죠? 국어 여행을 떠날 준비물, 교안을 어떻게 받을 수 있는지 선생님이 설명을 해줄게요. 
    (EBS)/(이비에스) 초등을 검색해서 들어가면요 첫화면이 이렇게 나와요. 
    자/ 그러면요 여기 (X)/(엑스) 눌러주(고요)/(구요). 저기 (동그라미)/(똥그라미) (EBS)/(이비에스) (2주)/(이 주) 라이브특강이라고 되어있죠? 
    거기를 바로 가기를 누릅니다. 자/ (누르면요)/(눌르면요). 어떻게 되냐? b/ 밑으로 내려요 내려요 내려요 쭉 내려요. 
    우리 몇 학년이죠? 아/ (2학년)/(이 학년) 이죠 (2학년)/(이 학년)의 무슨 과목? 국어. 
    이번주는 (1주)/(일 주) 차니까요 여기 교안. 다음주는 여기서 다운을 받으면 돼요. 
    이 교안을 클릭을 하면, 짜잔/. 이렇게 교재가 나옵니다 .이 교안을 (다운)/(따운)받아서 우리 국어여행을 떠날 수가 있어요. 
    그럼 우리 진짜로 국어 여행을 한번 떠나보도록 해요? 국어여행 출발. 자/ (1단원)/(일 단원) 제목이 뭔가요? 한번 찾아봐요. 
    시를 즐겨요 에요. 그냥 시를 읽어요 가 아니에요. 시를 즐겨야 돼요 즐겨야 돼. 어떻게 즐길까? 일단은 내내 시를 즐기는 방법에 대해서 공부를 할 건데요. 
    그럼 오늘은요 어떻게 즐길까요? 오늘 공부할 내용은요 시를 여러 가지 방법으로 읽기를 공부할겁니다. 
    어떻게 여러가지 방법으로 읽을까 우리 공부해 보도록 해요. 오늘의 시 나와라 짜잔/! 시가 나왔습니다 시의 제목이 뭔가요? 다툰 날이에요 다툰 날. 
    누구랑 다퉜나 동생이랑 다퉜나 언니랑 친구랑? 누구랑 다퉜는지 선생님이 시를 읽어 줄 테니까 한번 생각을 해보도록 해요."""

inputs = [prefix + sample]


inputs = tokenizer(inputs, max_length=512, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=3, do_sample=True, min_length=10, max_length=64)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
result = nltk.sent_tokenize(decoded_output.strip())[0]

print('RESULT >>', result)

RESULT >> 국어 여행을 떠나기 전에 국어 여행을 떠날 준비물과 교안을 어떻게 받을 수 있는지 선생님이 설명해 준다.

```