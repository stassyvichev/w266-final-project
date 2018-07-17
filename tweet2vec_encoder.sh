#!/bin/bash

# w266 final project version - SMV
# call like this: ./tweet2vec_encoder.sh data/labeled_data_text_only.csv tweet2vec/tweet2vec/best_model/ data/encoded_data

infile="$1"
datafile="${infile}.preprocessed"
modelpath="$2" # "tweet2vec/tweet2vec/best_model/"
resultpath="$3"

echo "Inputs:"
echo $infile
echo $datafile
echo $modelpath
echo $resultpath

echo "Preprocessing:"
python tweet2vec/misc/preprocess.py $infile $datafile

echo "Create result dir"
mkdir -p $resultpath

echo "Run encoder"
python tweet2vec/tweet2vec/encode_char.py $datafile $modelpath $resultpath
echo "Done"
