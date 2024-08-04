import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
# Downloading necessary NLTK data for preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
# Load the dataset 
df = pd.read_csv('mail_data.csv')
print(df.head())
# Function to clean and preprocess the text data
def preprocess_text(text):
    # Converting text to lowercase
    text = text.lower()
    # Removing digits and special characters
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    # Removing extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenizing and remove stopwords
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
   
    return ' '.join(tokens)
# Applying the preprocessing to column 'Message' 
df['Message'] = df['Message'].apply(preprocess_text)

# Map the column 'Category'  to numerical values
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.2, random_state=42)
# Use TF-IDF to vectorize the text data
vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)


lr = LogisticRegression()
lr.fit(x_train_tfidf, y_train)

# Make predictions on the test set
y_pred = lr.predict(x_test_tfidf)
y_pred_proba = lr.predict_proba(x_test_tfidf)[:, 1]
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print(f'ROC-AUC: {roc_auc}')
# Ploting the ROC curve to visualize the model's performance
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(4, 4))
plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
# Count the occurrences of each class in the predictions
pred_counts = pd.Series(y_pred).value_counts()

# Plot a pie chart to show the distribution of predictions
labels = ['Non-Spam', 'Spam']
colors = ['lightpink', 'black']
explode = (0.1, 0)

plt.figure(figsize=(4, 4))
plt.pie(pred_counts, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=130)
plt.title('Distribution of Predicted Classifications')
plt.show()
