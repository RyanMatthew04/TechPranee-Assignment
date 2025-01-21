from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
import joblib
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import fbeta_score, make_scorer

app = FastAPI()

df = None
model = None
MODEL_PATH = "dt.pkl"

# Endpoint: Upload Dataset
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global df
    try:
        df = pd.read_csv(file.file)
        if not {'QualityScore','DefectRate','ProductionVolume','MaintenanceHours','DefectStatus'}.issubset(df.columns):
            raise HTTPException(status_code=400, detail="Dataset must contain QualityScore, DefectRate, ProductionVolume, MaintenanceHours, DefectStatus")
        return {"message": "Dataset uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Endpoint: Train Model
@app.post("/train")
async def train_model():
    global df, model
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a dataset first.")

    try:

        X=df[['QualityScore','DefectRate','ProductionVolume','MaintenanceHours']]
        y=df['DefectStatus']

        f2_scorer = make_scorer(fbeta_score, beta=2)

        param_grid = {
            'max_features': [None, 'sqrt', 'log2'],
            'ccp_alpha': [0.1, 0.01, 0.001],
            'max_depth': [3, 5, 7],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 5, 10],
            'random_state':[1]
        }

        strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        dt = DecisionTreeClassifier()

        grid_search = GridSearchCV(
            estimator=dt,
            param_grid=param_grid,
            scoring=f2_scorer,
            cv=strat_kfold,
            verbose=1,
        )

        grid_search.fit(X, y)

        model= DecisionTreeClassifier(**grid_search.best_params_)

        model.fit(X,y)

        joblib.dump(model,filename='dt.pkl')

        return {
            "Message": "Model Trained Successfully",
            "F2_Score": grid_search.best_score_
        }
   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error Training Model: {str(e)}")

# Request Body for Predictions
class PredictionRequest(BaseModel):
    QualityScore: float
    DefectRate: float
    ProductionVolume: float
    MaintenanceHours: float

# Endpoint: Predict
@app.post("/predict")
async def predict(request: PredictionRequest):
    global model

    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=400, detail="No trained model available. Please train the model first.")

    try:
        input_data = [[request.QualityScore, request.DefectRate, request.ProductionVolume, request.MaintenanceHours]]
        prediction = model.predict(input_data)
        confidence = max(model.predict_proba(input_data)[0])
        defect = "High" if prediction[0] == 1 else "Low"

        return {"Defect Status": defect, "Confidence": f'{round(confidence, 4)*100}%'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)