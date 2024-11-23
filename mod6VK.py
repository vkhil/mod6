# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
# Assuming the dataset is saved as 'ecommerce_reviews.csv'
df = pd.read_csv('ecommerce_reviews.csv')

# Data Preprocessing
# Define sentiment labels based on ratings
def assign_sentiment_label(rating):
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'

df['Sentiment'] = df['Rating'].apply(assign_sentiment_label)

# Remove reviews with less than 5 words
df['Word_Count'] = df['Review_Text'].apply(lambda x: len(str(x).split()))
df = df[df['Word_Count'] >= 5]

# Drop unnecessary columns and handle missing data
df = df.dropna(subset=['Review_Text', 'Rating'])

# One-Hot Encode Product Category
df = pd.get_dummies(df, columns=['Product_Category'])

# Feature Extraction: TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(df['Review_Text']).toarray()

# Combine TF-IDF features with One-Hot Encoded Product Category
X = np.hstack((X_tfidf, df.iloc[:, df.columns.str.startswith('Product_Category')].values))
y = df['Sentiment']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = rf.predict(X_test)

# Print Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=['Positive', 'Neutral', 'Negative'])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=['Positive', 'Neutral', 'Negative'], 
            yticklabels=['Positive', 'Neutral', 'Negative'], cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Error Analysis
# Get misclassified samples
test_data = pd.DataFrame({'Review_Text': df.iloc[X_test.indices]['Review_Text'], 
                          'True_Label': y_test, 
                          'Predicted_Label': y_pred})
misclassified = test_data[test_data['True_Label'] != test_data['Predicted_Label']]

# Display 5 Misclassified Samples
print("5 Misclassified Samples:")
print(misclassified.head(5))
