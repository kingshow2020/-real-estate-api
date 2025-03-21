from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

app = FastAPI()

# ✅ نموذج البيانات المطلوبة من المستخدم
class RealEstateInput(BaseModel):
    area: float
    deposit: float
    opening_price: float
    type: str
    city: str

# ✅ تحميل أو تدريب النموذج
model_path = "model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    # بيانات تدريب وهمية (تقدر تستبدلها بملف CSV)
    data = pd.DataFrame({
        "area": [500, 1000, 1500],
        "deposit": [100000, 200000, 300000],
        "opening_price": [1000000, 2000000, 3000000],
        "type": ["سكني", "تجاري", "سكني"],
        "city": ["الرياض", "جدة", "الدمام"],
        "sale_price": [1400000, 2700000, 4000000]
    })

    # تحويل النوع والمدينة إلى أرقام (ترميز بسيط)
    data["type_code"] = data["type"].astype("category").cat.codes
    data["city_code"] = data["city"].astype("category").cat.codes

    X = data[["area", "deposit", "opening_price", "type_code", "city_code"]]
    y = data["sale_price"]

    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, model_path)

# ✅ التنبؤ
@app.post("/predict")
def predict_price(data: RealEstateInput):
    type_code = pd.Series([data.type]).astype("category").cat.codes[0]
    city_code = pd.Series([data.city]).astype("category").cat.codes[0]

    input_data = [[
        data.area,
        data.deposit,
        data.opening_price,
        type_code,
        city_code
    ]]

    prediction = model.predict(input_data)[0]
    safe_zone = data.opening_price * 1.37
    is_safe = prediction <= safe_zone

    return {
        "predicted_price": round(prediction),
        "safe_zone_limit": round(safe_zone),
        "is_safe_zone": is_safe
    }
