import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(df.head())
print(df.shape)

print(df.isnull().sum())
print(df.describe())

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


features = ["Pclass","Sex","Age","Fare","AgeGroup","IsRich","FamilySize"]
target = "Survived"

df = df[features+ [target]]


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression



X = df[features]
y = df[target]

X_train , X_test, y_train, y_test = train_test_split(
    X,y, test_size= 0.2,random_state = 42
)

numeric_features = ["IsRich","FamilySize"]
categorical_features = ["Pclass","Sex","AgeGroup"]

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
    ("classifier", LogisticRegression())
])

model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)
print("Accuracy:",accuracy)

predictions = model.predict(X_test[:5])
print(predictions)

print(df.columns)