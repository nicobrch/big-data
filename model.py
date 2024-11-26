from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle
import etl

dataset = etl.read_csv('data/twitter_training.csv')
X = etl.vectorize_text(dataset['tweet'])
y = dataset['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

with open('./model/svm.pkl', 'wb') as model_file:
    pickle.dump(svm, model_file)

print("Model saved to ./model/svm.pkl")

lr = LogisticRegression(max_iter=5000, random_state=42)
lr.fit(X_train, y_train)

with open('./model/lr.pkl', 'wb') as model_file:
    pickle.dump(lr, model_file)

print("Model saved to ./model/lr.pkl")