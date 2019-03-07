from server.service.classification_service import predict
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html', the_title='Toxic Language Classifier')

@app.route('/classify', methods=['POST'])
def add_message():
    content = request.values.to_dict(flat=False)
    text = content['input_text']
    print(text)

    prediction = predict(text)
    return prediction # prediction should already be in json form

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
