# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import string

# %%
# Load data
df = pd.read_csv(r'C:\Users\Pakistan\Downloads\all_tweets.csv', encoding='latin-1', header=None)
df.columns = ['sentiment', 'id', 'date', 'query', 'text']

print("=" * 60)
print("STEP 1: LOAD DATA")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nSentiment value counts:")
print(df['sentiment'].value_counts().sort_index())
print(f"\nData types:")
print(df.dtypes)

# Show some sample data
print(f"\nSample rows:")
print(df.head(10))

# %%
# Check what sentiment values actually exist
print("\n" + "=" * 60)
print("STEP 2: ANALYZE SENTIMENT VALUES")
print("=" * 60)
unique_sentiments = df['sentiment'].unique()
print(f"Unique sentiment values: {sorted(unique_sentiments)}")
print(f"Number of unique sentiments: {len(unique_sentiments)}")

# %%
# Binary mapping - CORRECTED
print("\n" + "=" * 60)
print("STEP 3: SENTIMENT MAPPING")
print("=" * 60)

# First, let's see what we have before filtering
print("Before filtering:")
print(df['sentiment'].value_counts().sort_index())

# Option A: If you have 0-4 scale, remove neutral and map to binary
if df['sentiment'].isin([0.0, 1.0, 2.0, 3.0, 4.0]).all():
    print("\nDetected 5-class sentiment (0-4)")

    # Remove neutral (2.0)
    df = df[df['sentiment'] != 2.0]
    print(f"\nAfter removing neutral (2.0): {df.shape}")
    print(df['sentiment'].value_counts().sort_index())

    # Map to binary
    sentiment_map = {
        0.0: 0,
        1.0: 0,
        3.0: 1,
        4.0: 1
    }
    df['sentiment'] = df['sentiment'].map(sentiment_map)

# Option B: If you have 0 and 4 only (like Sentiment140)
elif df['sentiment'].isin([0, 4]).any():
    print("\nDetected binary sentiment (0 and 4)")
    df = df[df['sentiment'].isin([0, 4])]
    df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})

# Option C: Already binary (0 and 1)
elif df['sentiment'].isin([0, 1]).all():
    print("\nAlready binary (0 and 1)")
    # No mapping needed

print(f"\nAfter mapping: {df.shape}")
print("\nFinal sentiment distribution:")
print(df['sentiment'].value_counts())

# CRITICAL CHECK
if df['sentiment'].nunique() < 2:
    print("\n❌ ERROR: Only one class remaining!")
    print("This means your mapping or filtering removed one entire class.")
    print("Let's reload and try a different approach...")

# %%
# Preprocessing
print("\n" + "=" * 60)
print("STEP 4: TEXT PREPROCESSING")
print("=" * 60)


def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|https\S+|www\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())

    return text


df['cleaned_text'] = df['text'].apply(preprocess_text)

# Check for empty texts
empty_count = (df['cleaned_text'].str.strip() == '').sum()
print(f"Empty texts after cleaning: {empty_count}")

# Remove empty and NaN
df = df[df['cleaned_text'].str.strip() != '']
df = df.dropna(subset=['cleaned_text', 'sentiment'])

print(f"\nAfter removing empty texts: {df.shape}")
print("\nSentiment distribution after cleaning:")
print(df['sentiment'].value_counts())

# CRITICAL CHECK AGAIN
if df['sentiment'].nunique() < 2:
    print("\n❌ ERROR: Only one class after preprocessing!")
    print("One class was completely removed during text cleaning.")
else:
    print(f"\n✅ Good! We have {df['sentiment'].nunique()} classes")

# %%
# Only proceed if we have both classes
if df['sentiment'].nunique() >= 2:
    print("\n" + "=" * 60)
    print("STEP 5: TRAIN-TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'],
        df['sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment']
    )

    print(f'Training size: {len(X_train)}')
    print(f'Test size: {len(X_test)}')

    print("\nTrain sentiment distribution:")
    print(y_train.value_counts())

    print("\nTest sentiment distribution:")
    print(y_test.value_counts())

    # %%
    # Vectorization
    print("\n" + "=" * 60)
    print("STEP 6: VECTORIZATION")
    print("=" * 60)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF shape (train): {X_train_tfidf.shape}")
    print(f"TF-IDF shape (test): {X_test_tfidf.shape}")

    # %%
    # Train models
    print("\n" + "=" * 60)
    print("STEP 7: MODEL TRAINING")
    print("=" * 60)

    # Bernoulli Naive Bayes
    print("\n--- Bernoulli Naive Bayes ---")
    bnb = BernoulliNB()
    bnb.fit(X_train_tfidf, y_train)
    bnb_pred = bnb.predict(X_test_tfidf)

    print(f"Accuracy: {accuracy_score(y_test, bnb_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        bnb_pred,
        labels=[0, 1],
        target_names=['Negative', 'Positive'],
        zero_division=0
    ))

    # %%
    # Logistic Regression
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_tfidf, y_train)
    lr_pred = lr.predict(X_test_tfidf)

    print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        lr_pred,
        labels=[0, 1],
        target_names=['Negative', 'Positive']
    ))

    # %%
    # Linear SVM
    print("\n--- Linear SVM ---")
    svm = LinearSVC(max_iter=1000, random_state=42)
    svm.fit(X_train_tfidf, y_train)
    svm_pred = svm.predict(X_test_tfidf)

    print(f"Accuracy: {accuracy_score(y_test, svm_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        svm_pred,
        labels=[0, 1],
        target_names=['Negative', 'Positive']
    ))

    # %%
    # Model Comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    results = {
        'Bernoulli Naive Bayes': accuracy_score(y_test, bnb_pred),
        'Logistic Regression': accuracy_score(y_test, lr_pred),
        'Linear SVM': accuracy_score(y_test, svm_pred)
    }

    for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model}: {acc:.4f}")

else:
    print("\n❌ CANNOT PROCEED: Need at least 2 classes for classification")
    print("\nPlease check your data file. Your CSV might have:")
    print("1. All tweets with the same sentiment")
    print("2. Wrong column order")
    print("3. Wrong delimiter")
    print("\nTry printing the first 20 rows to inspect the raw data")