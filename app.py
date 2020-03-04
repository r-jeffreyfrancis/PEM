from flask import Flask,render_template
from flask_bootstrap import Bootstrap

import sentimentAnalysis

app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route("/")
def landing():
    return render_template('landing.html')
@app.route("/about")
def about():
    return render_template('about.html')
@app.route("/features")
def features():
    return render_template('features.html')
# @app.route("/group-conversation.svg")
# def groupconvo():
#     return render_template('group-conversation.svg')


if __name__ == "__main__":
    app.run(host='0.0.0.0')
