import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

def age_group(age):
    if age < 13:
        return "child"
    elif age < 20:
        return "teen"
    elif age < 60:
        return "adult"
    return "senior"

df["AgeGroup"] = df["Age"].fillna(df["Age"].mean()).apply(age_group)

df["IsRich"] = (df["Fare"] > 50).astype(int)

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

df["HasCabin"] = df["Cabin"].notnull().astype(int)



features = ["Pclass","Sex","AgeGroup","IsRich","FamilySize","Title","HasCabin"]
target = "Survived"

df = df[features+ [target]]

X = df[features]
y = df[target]

X_train , X_test, y_train, y_test = train_test_split(
    X,y, test_size= 0.2,random_state = 42
)

numeric_features = ["IsRich","FamilySize","HasCabin"]
categorical_features = ["Pclass","Sex","AgeGroup","Title"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers = [
        ("num",numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("classifier", LogisticRegression(max_iter=200))
])

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy:", model.score(X_test, y_test))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nSample probabilities:")
print(model.predict_proba(X_test[:5]))

print("\nSurvival rate by Sex:")
print(df.groupby("Sex")["Survived"].mean())

print("\nSurvival rate by Class:")
print(df.groupby("Pclass")["Survived"].mean())

print("\nSurvival rate by AgeGroup:")
print(df.groupby("AgeGroup")["Survived"].mean())

sns.countplot(x="Sex", hue="Survived", data=df)
plt.show()

