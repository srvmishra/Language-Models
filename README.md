# Language-Models
This repository contains implementation of deep learning based language models using the PyTorch library starting with transformer based models. 

## HuggingFace Applications
### Text classification
1. Sentiment Classification in tweets using [Emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset, [DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased) model.
   - Description:
     - Dataset has 6 emotion categories and each tweet is tagged with a single emotion. Multi-Class Classification Problem.
     - Used random forests, svm, and logistic regression classifiers from scikit-learn on features extracted using pretrained DistilBERT base uncased model.
     - Later, finetuned DistilBERT base uncased.
   - Associated file: `text_classification_emotions.ipynb`.
   - Key Results:
   - HuggingFace Link: [emotions-dataset-distilbert-base-uncased](https://huggingface.co/srvmishra832/emotions-dataset-distilbert-base-uncased).
   - Reference: Chapter 2 [[1]](#1).
2. Similar vs Dissimilar Sentences: [MRPC dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52398) from [GLUE benchmark](https://gluebenchmark.com/), [Bert Base Uncased](https://huggingface.co/bert-base-uncased) model.
   - Description:
     - Dataset has pairs of sentences labeled as equivalent (similar) vs not equivalent (dissimilar). Binary Classification Problem - detecting whether sentences are paraphrases of each other.
     - Tokenized both sentences together, extracted features from pretrained BERT Base Uncased model. Used these features for binary classification with random forests, svm, and logistic regression classifiers from scikit-learn.
     - Tokenized each sentence separately and extracted features for each sentence using BERT Base Uncased model. Computed cosine similarity between these features on the training set and found a threshold for maximum F1 score. Applied this threshold for final classification on test set.
     - Later, finetuned BERT Base Uncased model on both sentences tokenized together.
   - Associated file: `BERT_Base_Uncased_Finetuning_on_GLUE_MRPC.ipynb`
   - Key Results:
   - HuggingFace Link: [glue-mrpc-bert-base-uncased ](https://huggingface.co/srvmishra832/glue-mrpc-bert-base-uncased)
   - Reference: Chapter 3 [[2]](#2).


## Transformers from scratch

1. Data Processing: (Chapter 2) [[3]](#3).
   - Implemented simple tokenizer using regular expressions.
   - Used the `tiktoken` library to tokenize texts
   - Created dataloaders for next token prediction task (data: `data/the-verdict.txt`)
   - Associated files: `Data_Processing.ipynb` and `GPT2_utils/data_processing.py` 
2. Building Blocks of GPT2 Model: (Chapters 3 and 4) [[3]](#3).
   - Implemented the self attention mechanism: `GPT2_utils/self_attention_mechanisms.py`
   - Other components: LayerNorm, GELU activation, FeedForwardNetwork, TransformerBlocks: `GPT2_utils/GPT_model_blocks.py`
   - Associated files: `GPT2_Model_from_scratch.ipynb`
3. Training GPT2 Model from scratch: (Chapter 5) [[3]](#3).
   - Training and Evaluation utilities: `GPT2_utils/train_and_eval_utils.py`
   - Downloading pretrained GPT2 weights: `GPT2_utils/gpt_download.py`, taken from: [Chapter 5 code](https://github.com/rasbt/LLMs-from-scratch/blob/2f41429cf422dd738903c342dc12b790a3e357d0/ch05/01_main-chapter-code/gpt_download.py)
   - Training data: `data/the-verdict.txt`, taken from: [Chapter 2 data](https://github.com/rasbt/LLMs-from-scratch/blob/2f41429cf422dd738903c342dc12b790a3e357d0/ch02/01_main-chapter-code/the-verdict.txt)
   - Associated files: `GPT2_Training_from_Scratch.ipynb`

To do: Implement Cross attention, transformer encoder and decoder blocks, PyTorch models (LSTM, GRU), finetuning GPT2 models for classification and instruction following tasks.


## References
<a id="1">[1]</a>. 
[Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/), O'Reilly Media, May 2022. 

<a id="2">[2]</a>.
[Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/)

<a id="3">[3]</a>
[Building a Large Language Model from Scratch](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167?crid=228R4JI0P0QFR&dib=eyJ2IjoiMSJ9.XvZyIer9iV133BWXqNiVt_OOJXZheO54dvZtQly8MC25PNYZrN3OWsGLjbg3I0G9hI3LkjwhsORxvHIob3nvCZFgdSSQEFe07VkehijGxT03n4Amdw7lnXxnsOUuWXeglfHnewCcV3DjL9zWHELfh5DG1ZErzFym3S6ZxSuFzNvoPkaq0uDlD_CKwqHdC0KM_RdvIqF0_2RudgvzRli0V155KkusHRck3pG7ybp5VyqKDC_GgL_MEywLwLhFgX6kOCgV6Rq90eTgSHFd6ac8krpIYjsHWe6H3IXbfKGvMXc.473O1-iUZC0z2hdx8L5Z5ZTNxtNV9gNPw_mE7QZ5Y90&dib_tag=se&keywords=raschka&qid=1730250834&sprefix=raschk,aps,162&sr=8-1&linkCode=sl1&tag=rasbt03-20&linkId=84ee23afbd12067e4098443718842dac&language=en_US&ref_=as_li_ss_tl)
