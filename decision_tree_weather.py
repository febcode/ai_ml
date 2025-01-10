from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

# Data (Weather Features and Decision)
weather = ['sunny and windy','winter and snow', 'rainy and thunder' , 'snow and clear' , 'shower and rainbow' , 'winter and sunny']  # Features (Sunny, Windy)
y = [0, 0, 0, 1,1 , 1]  # Labels (Play: Yes=1, No=0)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(weather)

# Model Training
model = DecisionTreeClassifier()
model.fit(X, y)

# Prediction


if (new_weather := ['rainy and thunder']):
    new_weather_vectorized = vectorizer.transform(new_weather)
    prediction = model.predict(new_weather_vectorized)
    # print(f"The email is {'Spam' if prediction[0] == 1 else 'Not Spam'}")
    print(f"Decision: {'Play' if prediction[0] == 1 else 'Don’t Play'}")
