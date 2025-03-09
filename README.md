# Language-Models
This repository contains implementation of deep learning based language models using the PyTorch library starting with transformer based models. 

Added:
1. HuggingFace Applications
   - Text classification
     - Dataset: [Emotion](https://huggingface.co/datasets/dair-ai/emotion),
     - Model: [DistilBERTt-base-uncased](https://huggingface.co/distilbert-base-uncased)
     - Description: Used random forests, svm, and logistic regression classifiers from **scikit-learn** on features extracted using pretrained DistilBERT base uncased model. Later, finetuned **DistilBERT base uncased** on **Emotion** dataset.
     - Associated file: `text_classification_emotions.ipynb`.
     - HuggingFace Link: [emotions-dataset-distilbert-base-uncased](https://huggingface.co/srvmishra832/emotions-dataset-distilbert-base-uncased).
     - Reference: Chapter 2 [[1]](#1)


Implemented self attention
To do: Implement Cross attention, transformer encoder and decoder blocks, PyTorch models (LSTM, GRU)

Results:

References:
<a id="1">[1]</a>. [Natural Language Processing with Transformers]((https://www.oreilly.com/library/view/natural-language-processing/9781098136789/)), O'Reilly Media, May 2022. 
