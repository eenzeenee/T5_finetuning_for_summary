# T5_finetuning_for_summary

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

