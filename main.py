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
        "predicted_price": f"{round(prediction):,} ريال",
        "safe_zone_limit": f"{round(safe_zone):,} ريال",
        "is_safe_zone": "✅ آمن" if is_safe else "⚠️ غير آمن"
    }
