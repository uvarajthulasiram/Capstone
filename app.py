from flask import Flask, render_template, request
from model import predict_sentiment, recommend_user

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    recommendations = []

    if request.method == "POST":
        review_text = request.form.get("review")
        username = request.form.get("username")

        if review_text:
            sentiment = predict_sentiment(review_text)

        if username:
            recommendations = recommend_user(username)

    return render_template(
        "index.html",
        sentiment=sentiment,
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run(debug=True)
