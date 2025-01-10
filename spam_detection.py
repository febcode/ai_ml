from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = [
    "Win a free lottery now", 
    "Important meeting tomorrow", 
    "You have won a prize", 
    "Schedule your doctor's appointment"
]
labels = [1, 0, 1, 0]  # Spam/Not Spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

if (new_email := ["Claim your free prize today"]):
    new_email_vectorized = vectorizer.transform(new_email)
    prediction = model.predict(new_email_vectorized)
    print(f"The email is {'Spam' if prediction[0] == 1 else 'Not Spam'}")
