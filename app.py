from flask import Flask, render_template, request
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd
import string

# Initialize Flask app
app = Flask(__name__)


#nltk.download('stopwords')


# Load stopwords from NLTK
#stop_words = set(nltk.corpus.stopwords.words('english'))


df=pd.read_csv('Financed_Labeled.csv')
df = df.dropna()
df = df[['Headlines', 'Description', 'Description_Sentiment']]

# Text Preprocessing Functions
def remove_punct(text):
    """Remove punctuation from text."""
    return ''.join([char for char in text if char not in string.punctuation])

def remove_tags(text):
    """Remove specific tags (like new lines, apostrophes) from text."""
    tags = ['\n', '\'']
    for tag in tags:
        text = text.replace(tag, '')
    return text

#def remove_stopwords(text):
 #   """Remove stopwords from the text."""
  #  words = nltk.word_tokenize(text)
   # filtered_words = [word for word in words if word.lower() not in stop_words]
    #return ' '.join(filtered_words)

def preprocess_text(text):
    """Apply all preprocessing functions in sequence."""
    text = remove_tags(text)
    text = remove_punct(text)
    #text = remove_stopwords(text)
    return text



X = df['Description']
y = df['Description_Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=212)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', LinearSVC())
])

pipeline.fit(X_train, y_train)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    description = request.form['description']
    preprocessed_description = preprocess_text(description)
    sentiment_prediction = pipeline.predict([preprocessed_description])[0]
    
    return render_template('index.html', description=description, sentiment=sentiment_prediction)


if __name__ == '__main__':
    app.run(debug=True)
