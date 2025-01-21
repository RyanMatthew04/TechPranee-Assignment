# Predictive Analysis for Manufacturing Operations

## Project Overview

This project focuses on predictive analysis to identify defect statuses in manufacturing operations. Using machine learning, the goal is to build a robust model that predicts whether the defect status of a manufacturing process will be **low (0)** or **high (1)**. Accurate predictions can help optimize operations, reduce defect rates, and improve overall production efficiency.

---

## Dataset

The dataset used for this project is sourced from Kaggle: [Predicting Manufacturing Defects Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset). It includes features related to various manufacturing parameters and their corresponding defect statuses.

- **Target Variable**: `DefectStatus`  
  - `0`: Low defect level  
  - `1`: High defect level  

---

## Thought Process

### 1. **Exploratory Data Analysis (EDA)**  
   - Performed EDA using the **`ydata_profiling`** library to gain insights into the dataset, including feature distributions, missing values, correlations, and other patterns.  
   - This step helped identify key trends and potential preprocessing needs.

### 2. **Data Preprocessing**  
   - Since the model selected was a Decision Tree, **no scaling or one-hot encoding** was required. Decision trees are inherently robust to different feature scales and can handle categorical data effectively.  

### 3. **Feature Selection**  
   - Leveraged the **feature importance** attribute of Decision Trees to select the most relevant features.  
   - Features with an importance score greater than **0.1** were retained for training, ensuring a balance between model complexity and interpretability.

### 4. **Model Training**  
   - Used **Grid Search** with **Stratified K-Fold Cross-Validation** to optimize hyperparameters and prevent overfitting.  
   - The **F2 Score** was chosen as the evaluation metric.  
     - **Reason**: In the context of manufacturing operations, reducing both false positives and false negatives is crucial to minimize waste and ensure quality control. However, reducing **false negatives** (predicting low defect levels when defects are actually high) is of higher priority. The F2 score places more emphasis on recall, aligning with this objective.

---

## Key Results

- Model selection and hyperparameter tuning using F2 score optimization.  
- Feature importance-driven feature selection ensured a focused and efficient training process.  
- Final model achieved an optimal balance between recall and precision, addressing the business problem effectively.

---

## Usage

1. Clone the repository:  
   ```bash
   git clone https://github.com/username/predictive-manufacturing-analysis.git

2. Create a new environment with Python 3.9.21:
   ```bash
   conda create -n manufacturing-env python=3.9.21
   conda activate manufacturing-env

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the api.py script:
   ```bash
   python api.py

5. API Interaction with Postman:
   
  ### 1. **Upload the Dataset**
  
    -Send a POST request to: http://0.0.0.0:8000/upload
    -In Postman, go to the Body tab and select form-data.
    -Set the key to file, input type to File, and upload the dataset as the value.
    -Send the request. It should return:
    {
      "message": "Dataset uploaded successfully"
    }

  ### 1. **Train the Model**
  
    -Send a POST request to: http://0.0.0.0:8000/train
    -In Postman, go to the Body tab and select none.
    -Send the request. It should return:
    {
      "Message": "Model Trained Successfully",
      "F2_Score": 0.9874629553566372
    }

  ### 1. **Test the Model Predictions**
  
    -Send a POST request to: http://0.0.0.0:8000/predict
    -In Postman, go to the Headers tab and set the key to Content-Type and value to application/json.
    -In the Body tab, select raw and enter the input data in the form of JSON:
    {
      "QualityScore": 63.46,
      "DefectRate": 3.12,
      "ProductionVolume": 202,
      "MaintenanceHours": 9
    }
    -Send the request. It should return:
    {
    "Defect Status": "High",
    "Confidence": "96.94%"
    }




  


