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

Implemented self attention

To do: Implement Cross attention, transformer encoder and decoder blocks, PyTorch models (LSTM, GRU)


## References
<a id="1">[1]</a>. 
[Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/), O'Reilly Media, May 2022. 

<a id="2">[2]</a>.
[Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/)
