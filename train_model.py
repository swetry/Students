# train_model.py
# =============================== #
# 1. Импорты
# =============================== #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.regularizers import l2

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# =============================== #
# 2. Загрузка данных
# =============================== #

# Заменить на твой файл
DATA_PATH = "students.csv"  # например

df = pd.read_csv(DATA_PATH)
print("=" * 80)
print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# =============================== #
# 3. Предобработка (как в твоём коде)
# =============================== #

def basic_preprocessing(df: pd.DataFrame):
    print("=" * 80)
    print("Basic preprocessing started")

    df_processed = df.copy()
    original_columns = df_processed.columns.tolist()

    # 3.1. Обработка выбросов для числовых колонок
    numeric_cols = ["age", "studyhours", "classattendance", "sleephours"]
    for col in numeric_cols:
        if col in df_processed.columns:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
            print(f"{col}: [{lower_bound:.2f}, {upper_bound:.2f}]")

    print("Step 1: Outliers handled")

    # 3.2. Инженерия признаков
    if set(["studyhours", "classattendance"]).issubset(df_processed.columns):
        df_processed["studyefficiency"] = df_processed["studyhours"] * df_processed["classattendance"] / 100.0

    if set(["sleephours", "classattendance"]).issubset(df_processed.columns):
        df_processed["totalsleepquality"] = df_processed["sleephours"] * df_processed["classattendance"] / 100.0

    if set(["studyhours", "sleephours"]).issubset(df_processed.columns):
        df_processed["studysleepbalance"] = df_processed["studyhours"] / (df_processed["sleephours"] + 0.1)

    print("Step 2: Feature engineering done")

    # 3.3. Категориальные признаки
    categorical_cols = df_processed.select_dtypes(include="object").columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ["studentid", "examscore"]]

    binary_cols = []
    multi_cols = []
    for col in categorical_cols:
        nunique = df_processed[col].nunique()
        if nunique == 2:
            binary_cols.append(col)
        elif nunique > 2:
            multi_cols.append(col)

    print("Binary cols:", binary_cols)
    print("Multi cols:", multi_cols)

    # Label Encoding для бинарных
    for col in binary_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        print("Label Encoding:", col)

    # One-Hot Encoding для multi_cols
    for col in multi_cols:
        if col in df_processed.columns:
            unique_values = df_processed[col].unique()
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            df_processed = pd.concat([df_processed, dummies], axis=1)
            df_processed.drop(col, axis=1, inplace=True)
            print(f"One-Hot Encoding: {col}, orig={len(unique_values)}, new={dummies.shape[1]}")

    print("Step 3: Categorical encoding done")

    # 3.4. Масштабирование
    all_numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    numeric_to_scale = [c for c in all_numeric_cols if c not in ["studentid", "examscore"]]

    print("Numeric to scale:", len(numeric_to_scale))

    scaler = None
    if numeric_to_scale:
        scaler = StandardScaler()
        df_processed[numeric_to_scale] = scaler.fit_transform(df_processed[numeric_to_scale])
        print("StandardScaler applied")
    else:
        print("No numeric columns to scale")

    print("Step 4: Scaling done")

    # 3.5. Разделение на X, y
    X = df_processed.drop(["studentid", "examscore"], axis=1, errors="ignore")
    y = df_processed["examscore"]
    feature_names = X.columns.tolist()

    print("Final feature count:", X.shape[1])
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y, scaler, feature_names

X, y, scaler, feature_names = basic_preprocessing(df)

# =============================== #
# 4. Отбор признаков RandomForest
# =============================== #

print("=" * 80)
print("Feature selection with RandomForest")

X_temp, _, y_temp, _ = train_test_split(X, y, test_size=0.3, random_state=42)

rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
rf_selector.fit(X_temp, y_temp)

importancedf = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_selector.feature_importances_
}).sort_values("importance", ascending=False)

print("Top 15 features:")
print(importancedf.head(15).to_string(index=False))

N_FEATURES = min(20, len(feature_names))
selected_features = importancedf.head(N_FEATURES)["feature"].tolist()

print("Selected features count:", len(selected_features))
print(selected_features)

X_selected = X[selected_features]

# =============================== #
# 5. Train/test split
# =============================== #

print("=" * 80)
print("Train/test split")

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, shuffle=True
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# =============================== #
# 6. Бенчмарки классических моделей
# =============================== #

print("=" * 80)
print("Benchmark models")

benchmark_models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
}

benchmark_results = {}

for name, model in benchmark_models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    benchmark_results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "model": model}
    print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

# =============================== #
# 7. Нейросеть
# =============================== #

def build_optimized_nn(input_dim: int) -> keras.Model:
    print("Building optimized NN, input_dim:", input_dim)
    model = Sequential(name="OptimizedExamPredictor")
    model.add(Input(shape=(input_dim,)))
    # 1
    model.add(Dense(128, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    # 2
    model.add(Dense(64, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    # 3
    model.add(Dense(32, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    # 4
    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))
    # output
    model.add(Dense(1, activation="linear"))

    optimizer = Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="huber", metrics=["mae", "mse"])
    return model

input_dim = X_train.shape[1]
nn_model = build_optimized_nn(input_dim)
print(nn_model.summary())

callbacks = [
    EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1),
    ModelCheckpoint("bestnnmodel.keras", monitor="val_loss", save_best_only=True, verbose=0),
]

history = nn_model.fit(
    X_train,
    y_train,
    validation_split=0.15,
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

y_pred_nn = nn_model.predict(X_test).flatten()
mae_nn = mean_absolute_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
r2_nn = r2_score(y_test, y_pred_nn)

print(f"Neural Network: MAE={mae_nn:.2f}, RMSE={rmse_nn:.2f}, R2={r2_nn:.4f}")

benchmark_results["Neural Network"] = {
    "MAE": mae_nn,
    "RMSE": rmse_nn,
    "R2": r2_nn,
    "model": nn_model,
}

# =============================== #
# 8. Сохраняем лучшую модель и метаданные
# =============================== #

# выбираем модель с минимальным MAE
best_name, best_info = min(benchmark_results.items(), key=lambda x: x[1]["MAE"])
best_model = best_info["model"]

print("=" * 80)
print("Best model:", best_name)
print(f"MAE={best_info['MAE']:.2f}, RMSE={best_info['RMSE']:.2f}, R2={best_info['R2']:.4f}")

# сохраняем scaler и selected_features
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selected_features, "selectedfeatures.pkl")

# сохраняем модель
if "keras" in str(type(best_model)).lower():
    best_model.save("bestexampredictor.keras")
    print("Saved bestexampredictor.keras")
else:
    joblib.dump(best_model, "bestexampredictor.pkl")
    print("Saved bestexampredictor.pkl")

metadata = {
    "best_model": best_name,
    "best_mae": float(best_info["MAE"]),
    "best_rmse": float(best_info["RMSE"]),
    "best_r2": float(best_info["R2"]),
    "features_count": len(selected_features),
    "features": selected_features,
    "train_shape": X_train.shape,
    "test_shape": X_test.shape,
}

with open("modelmetadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print("Saved modelmetadata.json")
print("=" * 80)
print("Training finished.")
