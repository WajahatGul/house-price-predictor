from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd
# Load scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

# Conversion factor
MARLA_TO_SQFT = 272.25
USD_TO_PKR = 280  # Or current exchange rate

def home(request):
    return render(request, "home.html")

def predict(request):
    if request.method == 'POST':
        try:
            # Get marla-based inputs
            lotarea_marla = float(request.POST.get('lotarea_marla', 10))
            grlivarea_marla = float(request.POST.get('grlivarea_marla', 8))
            basement_marla = float(request.POST.get('basement_marla', 5))
            overallqual = int(request.POST.get('OverallQual', 5))
            garagecars = int(request.POST.get('GarageCars', 1))

            # Convert marlas to sq ft
            LotArea = lotarea_marla * MARLA_TO_SQFT
            GrLivArea = grlivarea_marla * MARLA_TO_SQFT
            TotalBsmtSF = basement_marla * MARLA_TO_SQFT

            # Prepare input array
            input_df = pd.DataFrame([{
                'LotArea': LotArea,
                'OverallQual': overallqual,
                'GrLivArea': GrLivArea,
                'GarageCars': garagecars,
                'TotalBsmtSF': TotalBsmtSF
            }])
            input_scaled = scaler.transform(input_df)

            prediction_usd = model.predict(input_scaled)[0]

            # Convert to PKR and format with commas
            prediction_pkr = prediction_usd * USD_TO_PKR
            formatted_prediction = f"Rs. {int(prediction_pkr):,}"
            print("Prediction:", formatted_prediction)
            return render(request, 'home.html', {'prediction': formatted_prediction})
        
        except Exception as e:
            return render(request, 'home.html', {'error': str(e)})

    return render(request, 'home.html')
