from flask import Flask, request, jsonify
import logging
import pickle
import time
import etl

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

dataset = etl.read_csv('data/twitter_training.csv')
vectorizer = etl.TfidfVectorizer()
vectorizer.fit(dataset['tweet'])

with open('./model/svm.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

with open('./model/lr.pkl', 'rb') as file:
    lr_model = pickle.load(file)

@app.route('/svm', methods=['POST'])
def svm():
    start_time = time.time()
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'})
    tweet = data['text']
    cleaned_tweet = etl.clean_text(tweet)
    vectorized_tweet = vectorizer.transform([cleaned_tweet])
    prediction = svm_model.predict(vectorized_tweet)
    response_time = time.time() - start_time
    logging.info(f"Prediction: {prediction[0]}, Response time: {response_time:.4f} seconds.")
    return jsonify({
        'prediction': int(prediction[0]),
        'response_time': round(response_time, 4)
    })

@app.route('/lr', methods=['POST'])
def lr():
    start_time = time.time()
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'})
    tweet = data['text']
    cleaned_tweet = etl.clean_text(tweet)
    vectorized_tweet = vectorizer.transform([cleaned_tweet])
    prediction = lr_model.predict(vectorized_tweet)
    response_time = time.time() - start_time
    logging.info(f"Prediction: {prediction[0]}, Response time: {response_time:.4f} seconds.")
    return jsonify({
        'prediction': int(prediction[0]),
        'response_time': round(response_time, 4)
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)