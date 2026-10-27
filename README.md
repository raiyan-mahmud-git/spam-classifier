# SMS Spam Classifier

A machine learning project that classifies SMS messages as **spam** or **ham** (not spam) using TF-IDF features and a Multinomial Naive Bayes model. The model is tuned with cross-validation and evaluated on a held-out test set.

## Results

Trained on 80% of the [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,572 messages, 13% spam) and tested on the other 20% (1,115 messages) that was never used during tuning.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| ham | 0.986 | 0.998 | 0.992 |
| spam | 0.985 | 0.906 | 0.944 |

Overall accuracy: **98.6%**. Cross-validated spam F1 on the training data: **0.956**.

Confusion matrix on the test set:

| | Predicted ham | Predicted spam |
|---|---|---|
| **Actual ham** | 964 | 2 |
| **Actual spam** | 14 | 135 |

Only 2 legitimate messages out of 966 were wrongly flagged as spam, while 14 of the 149 spam messages slipped through. For a spam filter, blocking real messages is usually the worse mistake, so this trade-off is reasonable.

## Why not just accuracy?

Only about 13% of the messages are spam, so a model that always answered "ham" would already score about 87% accuracy. That is why the model is tuned and judged on spam precision, recall and F1.

## How it works

1. Load the dataset (downloaded once and cached in `data/`).
2. Split into train and test sets, keeping the spam ratio the same in both (stratified split).
3. Build one scikit-learn `Pipeline` of a TF-IDF vectorizer and a Naive Bayes model. Keeping both in one pipeline means the vectorizer only learns from training data, which avoids data leakage.
4. Tune the settings (stop words, 1-word vs 1-2 word phrases, minimum word frequency, smoothing) with 5-fold cross-validation, optimising spam F1.
5. Evaluate once on the untouched test set and save the model to `spam_model.joblib`.

Best settings found: word pairs included (`ngram_range=(1, 2)`), no stop-word removal, `alpha=0.01`.

## Setup

Requires Python 3.10 or newer.

```
python -m pip install -r requirements.txt
```

## Usage

Train and evaluate the model, then save it locally as `spam_model.joblib` (this generated model file is not tracked by Git):

```
python spam_classifier.py train
```

Training downloads and caches the dataset in `data/sms.tsv`. After training, classify your own messages:

```
python spam_classifier.py predict "Claim your free lottery prize now!" "See you at 6?"
```

Example output:

```
SPAM (100.0% spam)  Claim your free lottery prize now!
HAM  (  0.0% spam)  See you at 6?
```

## Limitations and next steps

- The best `alpha` landed at the edge of the search range, so trying even smaller values might help.
- Recall on spam is 90.6%. Adding character-level features (digits, currency symbols, capital letters) or trying Logistic Regression or a linear SVM could catch more of the missed spam.
- The dataset is small and fairly old, so the model may do worse on modern spam.


## Dataset

SMS Spam Collection (Almeida and Hidalgo), loaded from the `justmarkham/pycon-2016-tutorial` repository.
