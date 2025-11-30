import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import urllib.request
import io
import zipfile

print("Downloading dataset...")

# Load SMS Spam Collection dataset
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
response = urllib.request.urlopen(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
data = zip_file.read("SMSSpamCollection").decode("utf-8")

messages = []
labels = []

for line in data.split("\n"):
    parts = line.split("\t")
    if len(parts) == 2:
        labels.append(parts[0])
        messages.append(parts[1])

df = pd.DataFrame({"label": labels, "message": messages})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF + Naive Bayes
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB())
])

print("Training model...")
model.fit(X_train, y_train)

# Evaluation
pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# User input loop
print("\n--- SMS Spam Classifier ---")
while True:
    msg = input("Enter a message (or type exit): ")
    if msg.lower() == "exit":
        break
    print("Prediction:", model.predict([msg])[0])
