from flask import Flask,render_template, request
# from flask_bootstrap import Bootstrap

app = Flask(__name__)
# bootstrap = Bootstrap(app)
keyword = ""

@app.route("/")
def landing():
    return render_template('landing.html')
@app.route("/about")
def about():
    return render_template('about.html')
@app.route("/features")
def features():
    return render_template('features.html')
# @app.route("/run", methods=['POST'])
# def run():
#     form = request.form
#     if request.method == 'POST':
#         keyword = request.form['keyword']
#         import sentimentAnalysis
#         sentimentAnalysis.main(keyword)
#     return render_template('visualization.html')
@app.route("/result")
def result():
    return render_template('pre_loaded_vis.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0')
