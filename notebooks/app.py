import joblib

tfidf = joblib.load("tfidf.pkl")
svd = joblib.load("svd.pkl")
svm_clf = joblib.load("svm_classifier.pkl")
gbr = joblib.load("gbr_regressor.pkl")
import numpy as np
import re

def extract_numeric_features_single(text):
    text = text.lower()
    text_length = len(text)
    word_count = text.count(" ")
    math_symbol_count = len(re.findall(r"[+\-*/%=<>]", text))

    keywords = [
        "dp", "graph", "tree", "dfs", "bfs",
        "recursion", "bitmask", "greedy", "flow"
    ]

    keyword_counts = [text.count(kw) for kw in keywords]

    return np.array(
        [text_length, word_count, math_symbol_count] + keyword_counts
    ).reshape(1, -1)
def predict_difficulty(description, input_desc, output_desc):
    combined_text = f"{description} {input_desc} {output_desc}".lower()

    # TF-IDF
    X_tfidf = tfidf.transform([combined_text])

    # Numeric features
    X_num = extract_numeric_features_single(combined_text)

    # ---- Classification ----
    X_cls = np.hstack([X_tfidf.toarray(), X_num])
    pred_class = svm_clf.predict(X_cls)[0]

    # ---- Regression ----
    X_tfidf_red = svd.transform(X_tfidf)
    X_reg = np.hstack([X_tfidf_red, X_num])
    pred_score = gbr.predict(X_reg)[0]

    return pred_class, round(float(pred_score), 2)
import gradio as gr

iface = gr.Interface(
    fn=predict_difficulty,
    inputs=[
        gr.Textbox(lines=8, label="Problem Description"),
        gr.Textbox(lines=4, label="Input Description"),
        gr.Textbox(lines=4, label="Output Description")
    ],
    outputs=[
        gr.Label(label="Predicted Difficulty Class"),
        gr.Number(label="Predicted Difficulty Score")
    ],
    title="AutoJudge – Programming Problem Difficulty Predictor",
    description="Paste a problem statement to get difficulty class and score"
)

iface.launch()
