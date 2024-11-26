import re
import nltk
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text: str):
    text = re.sub(r'http\S+', '', text)  # Eliminar URLs
    text = re.sub(r'@[\w]*', '', text)  # Eliminar menciones (@usuario)
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Eliminar caracteres no alfabéticos
    text = text.lower() # Convertir a minúsculas
    words = text.split()    # Tokenizar
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [w for w in words if w not in stop_words]  # Eliminar stopwords
    lemmatizer = nltk.stem.WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]  # Lematizar
    return ' '.join(words)

def read_csv(path: str):
    df = pandas.read_csv(path)
    df.columns = ['id', 'hilo', 'class', 'tweet']
    df['tweet'] = df['tweet'].fillna('').apply(clean_text)
    class_map = {
        'Positive': 1,
        'Negative': -1,
        'Neutral': 0,
        'Irrelevant': 2
    }
    df['class'] = df['class'].map(class_map)
    return df

def vectorize_text(text: list):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(text)