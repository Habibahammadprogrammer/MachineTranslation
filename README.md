# Machine Translation on an english-german dataset 
## Project Description
This project involves building a machine translation model to translate text from one language to another. The model is evaluated using the ROUGE metric, which measures the quality of translations by comparing the overlap between the generated output and reference translations.

## Key Components
Dataset: english-french dataset
Model: There are 3 models  
   1. A basic Seq2Seq model built using Pytorch and LSTM cells
   2. A Transformer from scratch built using Pytorch
   3. A pre-trained Transformer (GPT2) from the huggingface library
Evaluation: ROUGE metrics for evaluating translation quality and BLEU-score for the Seq2Seq Model

## Requirements
Make sure you have the following installed:

1.Python 3.8 or higher
2.PyTorch (for model building and training)
3.Hugging Face transformers (if you're using pre-trained models)
4.ROUGE library,BLEU from NLTK library (for evaluation)
## Dataset 
  The Dataset is from kaggle and is an english-german dataset 
[downlaod dataset from kaggle](https://www.kaggle.com/datasets/devanshusingh/machine-translation-dataset-de-en)
## Pre-Processing 
for the transfomer from scratch and the Seq2Seq model I used the sentencepiece library in order to tokenize the sentence and remove unwanted tokens whilst adding special tokens( padding tokens, end of sentnece token etc)
for the pre-trained transformer , I created the tokenizer itself using chain from itertools library 
## Model Architecture  
  1. Seq2Seq Model
     This was a simple LSTM encoder-decoder architecture with an embedding layer and an early stopping class so that during training it would stop training if the loss is not getting any better
  2. Trasnformer from scratch
     Each layer in the transformer was built in its seperate class (positional encoding , encoder , self-attention based on the transfomer model from "Attention is all you need" infamous paper
  3. Pre-trained GPT2
     This was simply importing the GPT2 transformer from the hugging face library
## Model Training 
 2 for loops were created (one for the training and validation sets and one for the inference dataset) using tqdm library , the progress bar was created , number of epochs for training were 10 and early stopping was implemented. for the pre-trained model , since the task was finetuning ,only 3 epochs have been created. 
## Evaluation 
  I used the ROUGE library for the transformers and a BLEU score for the Seq2Seq model 
## Results 
Transformer from scratch ROUGE-Scores: 
ROUGE-1:0.617
ROUGE-2:0.388
ROUGE-L:0.603
Transformer Pre-trained ROUGE-Scores:
ROUGE-1:0.252
ROUGE-2:0.1188
ROUGE-L:0.1987
Seq2Seq Model BLEU-Score:
sentence-level BLEU Score:0.0150
corpus-lebel BLEU Score:0.0395
## Future Work 
1. Not use a simple Seq2Seq Model for a complicated task like machine translation since it hardly learns anything
2. Not use GPT-2 and look for a more tailored transformer for machine translation Tasks
3. Add More Metrics like accuracy measures and classification report.
## Acknowlegments 
1.[Kaggle Notebook](https://www.kaggle.com/code/hakim11/machine-translation-using-transformer-from-scratch)
2.[Attention is all you need paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

