from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RealEstateInput(BaseModel):
    area: float
    deposit: float
    opening_price: float

@app.post("/predict")
def predict_price(data: RealEstateInput):
    predicted_price = data.opening_price * 1.37
    return {"predicted_price": predicted_price}
#اختبار
