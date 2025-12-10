from sklearn.feature_extraction.text import TfidfVectorizer

def build_features(df):
    df["combined_text"] = df["resume_text"] + " " + df["job_text"]
    tfidf = TfidfVectorizer(max_features=500)
    X = tfidf.fit_transform(df["combined_text"]).toarray()
    return X, df["match_score"].values, tfidf
