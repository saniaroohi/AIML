import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Load your dataset
df = pd.read_csv("iris.csv")  # change to your actual filename

# 2. Drop the 'Id' column and separate features/target
X = df.drop(columns=["Id", "Species"])
y = df["Species"]

# 3. Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 5. Try different values of K
k_values = list(range(1, 11))
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    accuracies.append(acc)

# 6. Plot accuracy vs. K
plt.figure(figsize=(8,5))
plt.plot(k_values, accuracies, marker='o')
plt.title('Accuracy vs. K value')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# 7. Train final model with optimal K (e.g., k=3)
k_opt = 3
knn = KNeighborsClassifier(n_neighbors=k_opt)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 8. Accuracy and Confusion Matrix
print(f"KNN Accuracy (k={k_opt}):", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (k={k_opt})")
plt.show()
