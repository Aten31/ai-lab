import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# LOAD DATA

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)


#  FEATURE ENGINEERING 


def age_group(age):
    if age < 13:
        return "child"
    elif age < 20:
        return "teen"
    elif age < 60:
        return "adult"
    return "senior"

df["Age"] = df["Age"].fillna(df["Age"].mean())

df["AgeGroup"] = df["Age"].apply(age_group)
df["IsRich"] = (df["Fare"] > 50).astype(int)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["HasCabin"] = df["Cabin"].notnull().astype(int)
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)


# FEATURES / TARGET

features = [
    "Pclass",
    "Sex",
    "AgeGroup",
    "IsRich",
    "FamilySize",
    "HasCabin",
    "Title"
]

target = "Survived"

X = df[features]
y = df[target]


#  TRAIN TEST SPLIT (IMPORTANT)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# PREPROCESSING PIPELINE


categorical_features = ["Pclass", "Sex", "AgeGroup", "Title"]
numeric_features = ["IsRich", "FamilySize", "HasCabin"]

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])

preprocessor = ColumnTransformer([
    ("cat", categorical_pipeline, categorical_features),
    ("num", numeric_pipeline, numeric_features)
])



# MODEL PIPELINE


model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state = 42
    ))
])


#  TRAIN MODEL

model.fit(X_train, y_train)

encoder = model.named_steps["preprocessor"] \
              .named_transformers_["cat"] \
              .named_steps["encoder"]

encoded_cat_features = encoder.get_feature_names_out(categorical_features)

all_features = list(encoded_cat_features) + numeric_features

rf = model.named_steps["classifier"]


feature_importance = pd.DataFrame({
    "feature": all_features,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFEATURE IMPORTANCE:")
print(feature_importance)


#  EVALUATION

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC SCORE:")
print(roc_auc_score(y_test, y_proba))


# CROSS VALIDATION 

cv_scores = cross_val_score(
    model, X, y, cv=5, scoring="roc_auc"
)

print("\nCROSS VALIDATION AUC:")
print(cv_scores)
print("Mean AUC:", cv_scores.mean())
print("Std AUC:", cv_scores.std())
