# w266-final-project: Detecting flu prevalence through social media

This is the repo for our W266 Final Project (Authors: Stanimir Vichev, Shelly Hsu, Ryan Fitzgerald)

For running the baseline model, see `Baseline_Model.ipynb`.

For running the feedforward NN model, see `feed-forward-nn.ipynb`.

We have used the tweet2vec encoding (with some code changes) from `https://github.com/bdhingra/tweet2vec`. *Note*: for the encoding steps below, (2,3) you might need to install `theano`, `lasagna`, and some other libraries (see the errors that you get for guidance).

*Note*: the mean teacher model runs on tensorflow v1.2.1, it has issues with later versions.

For running the mean teacher model (adopted directly from `https://github.com/CuriousAI/mean-teacher`), you need to do the below:
1. Unzip the unlabeled data from the zip file: `unzip data/sentiment_text_only.csv.preprocessed.zip -d data/`
2. (Optional, some data is already present) Encode the unlabeled data using tweet2vec: `python tweet2vec/tweet2vec/encode_char_batch.py data/sentiment_text_only.csv.preprocessed tweet2vec/tweet2vec/best_model/ data/encoded_sentiment_data`. *Note*: Encoding all the data will take some time and around 6GB of space. Feel free to stop the script with `Ctrl+C` whenever you think you have enough data (each batch is about 4MB, we used 400MB for example).
3. (Optional, data is already present) Encode the labeled data using tweet2vec: `python tweet2vec/tweet2vec/encode_char.py data/labeled_data_text_only.csv.preprocessed tweet2vec/tweet2vec/best_model/ data/encoded_data`. This should be fairly quick as there isn't a lot of data.
4. To get the data in the right format for training, follow the steps in `prepare_data_for_mean_teacher.ipynb`. Adjust the unlabeled data loop depending on how many batches you ran in step 2. You can also adjust the test/train/dev split here. 
5. To run the actual training with mean teacher, run `cd mean-teacher; python train_tweets.py`


