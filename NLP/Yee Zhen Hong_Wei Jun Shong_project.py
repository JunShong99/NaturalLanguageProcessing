import re
from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords

##import dataset##
news_data = load_files(r"C:\Users\User\PycharmProjects\NLP\Yee Zhen Hong_Wei Jun Shong_data")
X, y = news_data.data, news_data.target


##text preprocessing##
documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

#converting text to number
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

##finding tf-idf
from sklearn.feature_extraction.text import TfidfTransformer

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

##train and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

##using random forest to train and test data
classifier = RandomForestClassifier(n_estimators=70, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

##evaluate performance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nResult:")
print(classification_report(y_test, y_pred))

print("\nAccurate Accuracy:")
print(accuracy_score(y_test, y_pred))
