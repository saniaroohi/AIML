# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv('titanic.csv')  

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

print("Summary Statistics:")
print(df.describe())  
df.hist(bins=20, figsize=(12, 8), color='skyblue')
plt.suptitle("Histograms of Numeric Features")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplots for Age and Fare")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']], hue='Survived')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()
fig = px.scatter(df, x='Age', y='Fare', color='Survived', title='Age vs Fare (Survival)')
fig.show()
