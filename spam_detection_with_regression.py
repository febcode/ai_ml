from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Data (Keyword frequency and Spam Label)
# X = [[2, 1], [1, 3], [5, 4], [0, 0]]  # Features
# y = [1, 1, 0, 0]  # Labels (Spam or Not)

emails = [
    "Win a free lottery now", 
    "Important meeting tomorrow", 
    "You have won a prize", 
    "Schedule your doctor's appointment"
]
labels = [1, 0, 1, 0]  # Spam/Not Spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Model Training
model = LogisticRegression()
model.fit(X, labels)


if (new_email := ["Claim your free prize today"]):
    new_email_vectorized = vectorizer.transform(new_email)
    prediction = model.predict(new_email_vectorized)
    print(f"The email is {'Spam' if prediction[0] == 1 else 'Not Spam'}")