import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

window = tk.Tk()
window.title = ("Financial News Sentiment")
window.geometry("500x400")

columns = ['sentence', 'sentiment']

df = pd.read_csv("data.csv", usecols=columns)

# preprocessing of dataset
def preprocessing(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]','', text)
    
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    tokens = [item for item in tokens if item not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatization = [lemmatizer.lemmatize(item) for item in tokens]
    
    return ' '.join(lemmatization)

df['sentence'] = df['sentence'].apply(preprocessing)
# print(df)

label = {'positive': 1, 'negative': 0, 'neutral': 2}

df['sentiment'] = df['sentiment'].replace(label)
print(df)

# sentiment is dependent on sentence
x = df['sentence']
y = df['sentiment']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

nb_classifier = MultinomialNB()
nb_classifier.fit(x_train_vec, y_train)

# prediction of trained data
x_pred = nb_classifier.predict(x_train_vec)

# Prediction of test data
y_pred = nb_classifier.predict(x_test_vec)

# Evaluate the model
print("Accuracy of train data: ", accuracy_score(y_train, x_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

def predict_sentiment():
    input_text = news_entry.get()
    preprocessed_news = preprocessing(input_text)
    vectorized = vectorizer.transform(list(preprocessed_news))
    prediction = nb_classifier.predict(vectorized)[0]
    if prediction == 0:
      print('Financial News : Negative')
    elif prediction == 2:
      print('Financial News: Neutral')
    else:
      print('Financial News: Positive')
    
    messagebox.showinfo("Sentiment prediction", f"The predicted sentiment is : {prediction}")

news_label = ttk.Label(window, text="Enter financial news : ")
news_label.pack()

news_entry = ttk.Entry(window, width=60)
news_entry.pack()

analyze_btn = ttk.Button(window, text="Analyze", command=predict_sentiment)
analyze_btn.pack()

window.mainloop()

# while True:
#   user_input = input("Enter a sentence to check Financial News: ")

#   preprocessed_input = preprocessing(user_input)

#   input_tfidf = vectorizer.transform(preprocessed_input)

#   prediction = nb_classifier.predict(input_tfidf)[0]

#   if prediction == 0:
#     print('Financial News : Negative')
#   elif prediction == 1:
#     print('Financial News: Positive')
#   else:
#     print('Financial News: Neutral')