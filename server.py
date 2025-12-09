# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import json
import os

from tensorflow import keras

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для локального фронта
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================== #
# Загрузка артефактов модели
# =============================== #

# Загрузка scaler и selected_features
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selectedfeatures.pkl")

# Загрузка модели (keras или sklearn)
model = None
if os.path.exists("bestexampredictor.keras"):
    model = keras.models.load_model("bestexampredictor.keras")
    is_keras = True
elif os.path.exists("bestexampredictor.pkl"):
    model = joblib.load("bestexampredictor.pkl")
    is_keras = False
else:
    raise RuntimeError("Модель bestexampredictor не найдена")

with open("modelmetadata.json", "r", encoding="utf-8") as f:
    modelmetadata = json.load(f)

# =============================== #
# Входная схема запроса
# =============================== #

class ExamInput(BaseModel):
    age: float
    studyhours: float
    classattendance: float
    sleephours: float
    internetaccess: str | None = None
    gender: str | None = None
    course: str | None = None
    sleepquality: str | None = None
    studymethod: str | None = None
    facilityrating: str | None = None
    examdifficulty: str | None = None

# =============================== #
# Предобработка одного примера
# =============================== #

def preprocess_student(data: ExamInput) -> np.ndarray:
    # Превращаем в DataFrame с одной строкой
    row = {
        "studentid": 0,  # фиктивный id, если он ожидался
        "age": data.age,
        "studyhours": data.studyhours,
        "classattendance": data.classattendance,
        "sleephours": data.sleephours,
        "internetaccess": data.internetaccess or "yes",
        "gender": data.gender or "male",
        "course": data.course or "b.tech",
        "sleepquality": data.sleepquality or "average",
        "studymethod": data.studymethod or "self-study",
        "facilityrating": data.facilityrating or "medium",
        "examdifficulty": data.examdifficulty or "moderate",
        "examscore": 0.0,  # заглушка, чтобы структура совпадала
    }
    df = pd.DataFrame([row])

    # те же инженерные признаки
    df["studyefficiency"] = df["studyhours"] * df["classattendance"] / 100.0
    df["totalsleepquality"] = df["sleephours"] * df["classattendance"] / 100.0
    df["studysleepbalance"] = df["studyhours"] / (df["sleephours"] + 0.1)

    # упрощённое кодирование категорий под предположение, что в train было так же
    # internetaccess: yes=1, no=0
    df["internetaccess"] = df["internetaccess"].map({"yes": 1, "no": 0}).fillna(1)

    # gender: male/female/other -> One-Hot (такие же названия колонок, как после train)
    for val in ["female", "other"]:
        col_name = f"gender_{val}"
        df[col_name] = (df["gender"] == val).astype(int)
    df.drop(columns=["gender"], inplace=True)

    # course: b.tech, b.sc, ba, diploma (пример кодирования)
    for val in ["b.sc", "ba", "diploma"]:
        col_name = f"course_{val}"
        df[col_name] = (df["course"] == val).astype(int)
    df.drop(columns=["course"], inplace=True)

    # sleepquality
    for val in ["average", "good"]:
        col_name = f"sleepquality_{val}"
        df[col_name] = (df["sleepquality"] == val).astype(int)
    df.drop(columns=["sleepquality"], inplace=True)

    # studymethod
    for val in ["coaching", "online videos", "group study"]:
        col_name = f"studymethod_{val}"
        df[col_name] = (df["studymethod"] == val).astype(int)
    df.drop(columns=["studymethod"], inplace=True)

    # facilityrating
    for val in ["medium", "high"]:
        col_name = f"facilityrating_{val}"
        df[col_name] = (df["facilityrating"] == val).astype(int)
    df.drop(columns=["facilityrating"], inplace=True)

    # examdifficulty
    for val in ["moderate", "hard"]:
        col_name = f"examdifficulty_{val}"
        df[col_name] = (df["examdifficulty"] == val).astype(int)
    df.drop(columns=["examdifficulty"], inplace=True)

    # теперь масштабируем числовые признаки тем же scaler
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_to_scale = [c for c in all_numeric_cols if c not in ["studentid", "examscore"]]

    df_scaled = df.copy()
    df_scaled[numeric_to_scale] = scaler.transform(df_scaled[numeric_to_scale])

    # выбираем только те признаки, которые использовались при обучении
    for col in selected_features:
        if col not in df_scaled.columns:
            df_scaled[col] = 0.0  # если какой‑то dummy‑признак не появился

    X_new = df_scaled[selected_features]
    return X_new.values  # shape (1, n_features)

# =============================== #
# Эндпоинт предсказания
# =============================== #

@app.post("/predict")
def predict_exam(input_data: ExamInput):
    try:
        X_new = preprocess_student(input_data)
        if model is None:
            raise RuntimeError("Модель не загружена")

        if "keras" in str(type(model)).lower():
            y_pred = model.predict(X_new, verbose=0).flatten()[0]
        else:
            y_pred = model.predict(X_new)[0]

        y_pred = float(y_pred)

        # простой текстовый комментарий на основе прогноза
        if y_pred < 50:
            comment = "Риск низкого результата. Модель советует усилить занятия и повысить посещаемость."
        elif y_pred < 75:
            comment = "Средний прогноз. При сохранении текущего режима есть шанс выйти на более высокий балл."
        else:
            comment = "Хороший прогноз. Важно поддерживать текущий баланс учёбы и сна."

        return {
            "prediction": y_pred,
            "comment": comment,
            "meta": {
                "model": modelmetadata.get("best_model", "unknown"),
                "mae": modelmetadata.get("best_mae", None),
                "r2": modelmetadata.get("best_r2", None),
            },
        }
    except Exception as e:
        return {"error": str(e)}

# для запуска: uvicorn server:app --reload --port 8000
