from flask import Flask, render_template, request, jsonify
import json
app = Flask(__name__)

# https://github.com/macloo/basic-flask-app
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html', the_title='Toxic Language Classifier')

@app.route('/classify', methods=['POST'])
def add_message():
    content = request.values.to_dict(flat=False)
    print(content['input_text'])
    print("HI")
    return jsonify({"predicted_label": "offensive", "confidence": 0.7343})

if __name__ == '__main__':
    app.run(debug=True)