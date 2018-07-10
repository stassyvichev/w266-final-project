#!/bin/bash

# w266 final project version - stan
# specify data file here
datafile="Colorado_Flu_Study_Tweets_Text_Only_Preprocessed.csv"

# specify model path here
modelpath="tweet2vec/tweet2vec/best_model/"

# specify result path here
resultpath="encoded_data/"

mkdir -p $resultpath

# test
python tweet2vec/tweet2vec/encode_char.py $datafile $modelpath $resultpath

