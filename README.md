# AgriSmart: Crop Yield Prediction & Plant Disease Detection

A powerful machine learning tool designed to empower farmers by providing data-driven insights for enhancing agricultural productivity. This project combines two predictive models into a single application:
1.  **Crop Yield Prediction:** Estimates yield productivity (in hg/ha) based on environmental and agricultural inputs.
2.  **Plant Disease Classification:** Identifies diseases from leaf images to enable early intervention.

## Features

- **Yield Estimation:** Predicts crop yield using key parameters like temperature, rainfall, and pesticide use. Powered by a highly accurate **RandomForest Regressor**, selected for its superior performance over other **ML algorithms** for this specific task.

- **Disease Diagnosis:** Accurately classifies plant health from leaf images across **38 different classes** using a state-of-the-art **EfficientNetB0** deep learning model, chosen after benchmarking against other pretrained networks.

- **User-Friendly Interface:** Offers both individual and combined intuitive interfaces for seamless interaction with both prediction models.

- **Optimized Models:** Utilizes competitively selected, state-of-the-art deep learning (**EfficientNetB0**) and machine learning (**RandomForest**) algorithms, ensuring best-in-class performance through a rigorous evaluation process.



## Results

### Plant Disease Classification (EfficientNetB0)
The model achieved exceptional accuracy on the Plant Village dataset:
-   **Accuracy:** 98.85%
-   **Precision:** 98.58%
-   **Recall:** 98.42%
-   **F1 Score:** 98.47%

### Crop Yield Prediction (RandomForest Regressor)
The yield prediction model demonstrated highly accurate results:
-   **R2 Score:** 0.974
-   **Mean Absolute Error (MAE):** 5706.45 hg/ha
-   **Root Mean Squared Error (RMSE):** 13528.55 hg/ha

## Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Ghifar-Khder/AgriSmart.git
    cd AgriSmart
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Setup

This project uses two separate datasets:

1.  **Crop Yield Prediction Dataset**
    -   Download from: [Kaggle - Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)
    -   The preprocessing code will split this into train/validation/test sets.

2.  **PlantVillage Dataset (for disease classification)**
    -   Download from: [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
    -   The preprocessing code will split this into train/validation/test sets.

**Important Setup Notes:**
-   The datasets are not stored in this repository due to their size.
-   After downloading the datasets, you must update all file paths in the code to match your local directory structure.
-   The preprocessing scripts will automatically create the train/validation/test splits.
-   Paths for saving models and results also need to be configured to your local environment.

## Usage

### Running the Combined Application (Recommended)
Launch the unified interactive interface to use both models:
```bash
python interface/combined_interface.py
