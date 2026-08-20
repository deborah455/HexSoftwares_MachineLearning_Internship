import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# Load cleaned data
df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = df[["v1", "v2"]].copy()
df.columns = ["label", "message"]
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1,
})


X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)


experiments = [
    {
        "name": "Baseline",
        "vectorizer": TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        ),
        "model": MultinomialNB(),
    },
    {
        "name": "Higher Alpha",
        "vectorizer": TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        ),
        "model": MultinomialNB(alpha=0.5),
    },
    {
        "name": "Lower Alpha",
        "vectorizer": TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        ),
        "model": MultinomialNB(alpha=0.1),
    },
    {
        "name": "Unigrams + Bigrams + Trigrams",
        "vectorizer": TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 3),
        ),
        "model": MultinomialNB(alpha=0.5),
    },
]


results = []

for experiment in experiments:

    vectorizer = experiment["vectorizer"]
    model = experiment["model"]

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    results.append({
        "Experiment": experiment["name"],
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
    })


results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print("=" * 75)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))