import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# --------------------------------
# COLUMN NAMES
# --------------------------------

columns = ["engine_id","cycle","op1","op2","op3"]

sensor_cols = ["sensor"+str(i) for i in range(1,22)]

all_columns = columns + sensor_cols


# --------------------------------
# FUNCTION TO LOAD DATASET
# --------------------------------

def load_dataset(path):
    
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1)
    df.columns = all_columns
    
    return df


# --------------------------------
# CALCULATE RUL FOR TRAIN DATA
# --------------------------------

def add_rul(df):

    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id","max_cycle"]

    df = df.merge(max_cycle,on="engine_id")

    df["RUL"] = df["max_cycle"] - df["cycle"]

    df.drop("max_cycle",axis=1,inplace=True)

    return df


# --------------------------------
# LOAD ALL TRAIN DATA
# --------------------------------

train_files = [
"data/train_FD001.txt",
"data/train_FD002.txt",
"data/train_FD003.txt",
"data/train_FD004.txt"
]

train_data = []

for file in train_files:

    df = load_dataset(file)
    df = add_rul(df)

    train_data.append(df)

train_data = pd.concat(train_data)


# --------------------------------
# TRAINING FEATURES
# --------------------------------

X_train = train_data.drop(["engine_id","cycle","RUL"],axis=1)
y_train = train_data["RUL"]


# --------------------------------
# NORMALIZATION
# --------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)


# --------------------------------
# TRAIN MODEL
# --------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("Training Model...")
model.fit(X_train,y_train)

print("Training Complete")


# --------------------------------
# TEST DATA
# --------------------------------

test_files = [
"data/test_FD001.txt",
"data/test_FD002.txt",
"data/test_FD003.txt",
"data/test_FD004.txt"
]

rul_files = [
"data/RUL_FD001.txt",
"data/RUL_FD002.txt",
"data/RUL_FD003.txt",
"data/RUL_FD004.txt"
]

all_predictions = []
all_actual = []


for test_file, rul_file in zip(test_files,rul_files):

    test_df = load_dataset(test_file)

    rul = pd.read_csv(rul_file,header=None)

    last_cycles = test_df.groupby("engine_id").last().reset_index()

    X_test = last_cycles.drop(["engine_id","cycle"],axis=1)

    X_test = scaler.transform(X_test)

    preds = model.predict(X_test)

    actual = rul[0].values

    all_predictions.extend(preds)
    all_actual.extend(actual)


# --------------------------------
# EVALUATION
# --------------------------------

rmse = np.sqrt(mean_squared_error(all_actual,all_predictions))
mae = mean_absolute_error(all_actual,all_predictions)
r2 = r2_score(all_actual,all_predictions)

print("\nModel Performance")
print("----------------------------")
print("RMSE:",rmse)
print("MAE:",mae)
print("R2 Score:",r2)


# --------------------------------
# VISUALIZATION
# --------------------------------

plt.figure(figsize=(6,6))
plt.scatter(all_actual,all_predictions,alpha=0.6)
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted RUL")


# --------------------------------
# FEATURE IMPORTANCE
# --------------------------------

feature_importance = model.feature_importances_

importance_df = pd.DataFrame({
"feature":sensor_cols + ["op1","op2","op3"],
"importance":feature_importance
})

importance_df = importance_df.sort_values("importance",ascending=False)


plt.figure(figsize=(8,6))
sns.barplot(x="importance",y="feature",data=importance_df)

plt.title("Feature Importance")


# --------------------------------
# PREDICTION ERROR DISTRIBUTION
# --------------------------------

errors = np.array(all_actual) - np.array(all_predictions)

plt.figure(figsize=(6,6))
sns.histplot(errors,bins=40,kde=True)

plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error")

plt.show()