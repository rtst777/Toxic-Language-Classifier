from flask import Flask, render_template, request, jsonify
import json
app = Flask(__name__)

# https://github.com/macloo/basic-flask-app


@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html', the_title='Toxic Language Classifier')

@app.route('/symbol.html')
def symbol():
    return render_template('symbol.html', the_title='Tiger As Symbol')

@app.route('/myth.html')
def myth():
    return render_template('myth.html', the_title='Tiger in Myth and Legend')

@app.route('/classify', methods=['POST'])
def add_message():
    content = request.values.to_dict(flat=False)
    print(content['input_text'])
    print("HI")
    return jsonify({"predicted_label": "offensive", "confidence": 0.77})

if __name__ == '__main__':
    app.run(debug=True)