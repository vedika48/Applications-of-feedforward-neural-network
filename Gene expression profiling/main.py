import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv(
    "/kaggle/input/datasets/vedikamali584/gene-expression/gene_expression.csv"
)

X = df.iloc[:, 2:].values
y = df["type"].values

encoder = LabelEncoder()
y = encoder.fit_transform(y)

num_classes = len(np.unique(y))

scaler = StandardScaler()
X = scaler.fit_transform(X)

pca = PCA(n_components=10)
X = pca.fit_transform(X)

def create_model():
    model = Sequential()
    model.add(Dense(3, input_dim=10, activation="sigmoid"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

loo = LeaveOneOut()

predictions = []
actual = []

sample_number = 1

for train_idx, test_idx in loo.split(X):

    print(f"Testing Sample {sample_number}/{len(X)}")

    X_train_full = X[train_idx]
    y_train_full = y[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    committee_votes = []

    for random_seed in range(100):

        kf = KFold(
            n_splits=8,
            shuffle=True,
            random_state=random_seed
        )

        for train_fold, val_fold in kf.split(X_train_full):

            X_train = X_train_full[train_fold]
            y_train = y_train_full[train_fold]

            model = create_model()

            model.fit(
                X_train,
                y_train,
                epochs=100,
                verbose=0
            )

            pred = model.predict(
                X_test,
                verbose=0
            )

            pred_class = np.argmax(pred)
            committee_votes.append(pred_class)

    final_prediction = Counter(
        committee_votes
    ).most_common(1)[0][0]

    predictions.append(final_prediction)
    actual.append(y_test[0])

    sample_number += 1

accuracy = accuracy_score(actual, predictions)

print("\nFinal Accuracy:", accuracy)

print("\nConfusion Matrix")
print(confusion_matrix(actual, predictions))

print("\nClassification Report")
print(
    classification_report(
        actual,
        predictions,
        target_names=encoder.classes_
    )
)