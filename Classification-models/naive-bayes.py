from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load breast cancer dataset
data = load_breast_cancer()

label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# ---- Data ------
# print("Breast Cancer data set Description :: ", data['DESCR'])
print(label_names)
# print(labels[0])
# print(feature_names)
# print(features)


# ------ Basic Split ------
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

# ------ Initialize -------
gnb = GaussianNB()

# ------ Train --------
model = gnb.fit(train, train_labels)

# ------ Predict ------
preds = gnb.predict(test)
# print(preds)

# ------ Accuracy ------
print(accuracy_score(test_labels, preds))

