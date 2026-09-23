# SIRRA

This repository contains the inference code for **SIRRA**, an open-domain table–text question answering pipeline evaluated on the OTT-QA benchmark.

## Directory structure

After downloading the required resources, the project directory should look like:

```text
.
├── log/
├── model/
│   ├── gte-multilingual-base/
│   ├── gte-multilingual-reranker-base/
│   ├── lora_question_to_table_reranker/
│   ├── lora_question_to_table_retriever/
│   ├── lora_reader/
│   ├── lora_row_scorer/
│   └── Qwen2.5-7B-Instruct/
├── ottqa_data/
│   ├── data/
│   ├── released_data/
│   ├── results/
│   ├── dev_test_possible_tables.pickle
│   ├── gte_multilingual_base_passages.faissindex
│   └── gte_multilingual_base_tables.faissindex
├── sirra/
│   └── utils/
└── main.py
```

`log/` is used to store runtime logs, and `ottqa_data/results/` is used to store inference results.

## Download resources

### 1. OTT-QA dataset

Download OTT-QA from:

https://github.com/wenhuchen/OTT-QA

Place the downloaded `data/` and `released_data/` directories under:

```text
ottqa_data/
```

### 2. Base models

Download the following models from Hugging Face and place them under `model/`:

- Alibaba-NLP/gte-multilingual-base  
  https://huggingface.co/Alibaba-NLP/gte-multilingual-base

- Alibaba-NLP/gte-multilingual-reranker-base  
  https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base

- Qwen/Qwen2.5-7B-Instruct  
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

### 3. LoRA adapters and FAISS indexes

Download the LoRA adapters, the precomputed FAISS indexes, and `dev_test_possible_tables.pickle` from:

https://pan.baidu.com/s/1Ed2MVIcEO624c-RRFlT2ug

Extraction code:

```text
odqa
```

Place the LoRA adapters under `model/`, and place the `.faissindex` files and `dev_test_possible_tables.pickle` under `ottqa_data/`, following the directory structure above.

## Run inference

Run on the OTT-QA development set:

```bash
python main.py dev
```

Run on the OTT-QA blind test set:

```bash
python main.py test
```

The inference results will be saved under:

```text
ottqa_data/results/
```
