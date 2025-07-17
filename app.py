from flask import Flask, render_template, request
import feedparser
import pickle
import os

# Load model and vectorizer
with open('models/fake_news_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    headlines = []

    if request.method == 'POST':
        action = request.form.get("action")

        if action == "predict_text":
            text = request.form.get("news_text")
            if text:
                vec = vectorizer.transform([text])
                pred = model.predict(vec)[0]
                conf = model.predict_proba(vec).max() * 100
                prediction = pred.upper()
                confidence = f"{conf:.2f}%"

        elif action == "predict_rss":
            rss_url = request.form.get("rss_url")
            if rss_url:
                feed = feedparser.parse(rss_url)
                for entry in feed.entries[:5]:  # limit to first 5 headlines
                    headline = entry.title
                    vec = vectorizer.transform([headline])
                    pred = model.predict(vec)[0]
                    conf = model.predict_proba(vec).max() * 100
                    headlines.append({
                        "headline": headline,
                        "prediction": pred.upper(),
                        "confidence": f"{conf:.2f}%"
                    })

        elif action == "predict_file":
            file = request.files.get("file")
            if file and file.filename.endswith('.txt'):
                text = file.read().decode('utf-8')
                vec = vectorizer.transform([text])
                pred = model.predict(vec)[0]
                conf = model.predict_proba(vec).max() * 100
                prediction = pred.upper()
                confidence = f"{conf:.2f}%"

    return render_template('index.html', prediction=prediction, confidence=confidence, headlines=headlines)

if __name__ == '__main__':
    app.run(debug=True)
