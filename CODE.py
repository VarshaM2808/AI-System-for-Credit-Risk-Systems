import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

#loading dataset
df = pd.read_csv(
    "german.data",
    sep=r"\s+",
    header=None
)

df.head(), df.shape
df.info()

print(df.columns)

columns = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age",
    "other_installment_plans", "housing", "existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker", "target"
]

df.columns = columns

print(df.columns)
df["target"].value_counts()
# 0 = good credit (low risk)
# 1 = bad credit (high risk)
df["risk"] = (df["target"] == 2).astype(int)

print(df["risk"].value_counts())


df["age_group"] = np.where(df["age"] < 25, "young", "older")

print(df["age_group"].value_counts())

X = df.drop(columns=["target", "risk"])
y = df["risk"]

X_encoded = pd.get_dummies(X, drop_first=True)

print("Feature matrix shape:", X_encoded.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred))

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: Credit Risk Prediction")
plt.show()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LogisticRegression(max_iter=3000)
model_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = model_scaled.predict(X_test_scaled)
cm_scaled = confusion_matrix(y_test, y_pred_scaled)
print(cm_scaled)

print(classification_report(y_test, y_pred_scaled))





