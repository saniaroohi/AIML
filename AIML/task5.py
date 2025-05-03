import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

data = pd.read_csv("heart.csv")  
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))

dot_data = export_graphviz(tree_clf, out_file=None,
                           feature_names=X.columns,
                           class_names=["No Disease", "Disease"],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("heart_tree")  
graph  

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

importances = rf_clf.feature_importances_
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.show()

cv_tree = cross_val_score(tree_clf, X, y, cv=5).mean()
cv_rf = cross_val_score(rf_clf, X, y, cv=5).mean()

print(f"Cross-Validated Accuracy (Decision Tree): {cv_tree:.4f}")
print(f"Cross-Validated Accuracy (Random Forest): {cv_rf:.4f}")
