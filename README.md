# Crop Disease Classification and Yield Prediction Platform

**App Link:** [https://agri4444.streamlit.app/](https://agri4444.streamlit.app/)

An integrated machine learning web application built with Streamlit for agricultural diagnostics and productivity forecasting. This platform combines deep learning computer vision for multi-class leaf disease identification with ensemble regression modeling for regional crop yield estimation.

---

## Technical Overview

The application unites two distinct machine learning pipelines into a single unified Streamlit interface:

* **Plant Pathology Diagnosis:** Classifies leaf health across **38 distinct crop and disease categories** using a fine-tuned **EfficientNetB0** model. Features an integrated **Grad-CAM heatmap visualizer** to display the spatial activation regions guiding each visual prediction.
* **Crop Yield Estimation:** Predicts agricultural productivity ($hg/ha$) based on regional climate factors (average rainfall, mean temperature) and management inputs (pesticide usage) using a **DecisionTree Regressor** pipeline.

---

## Model Evaluation & Performance

### Plant Disease Classification (EfficientNetB0)

Evaluated on the PlantVillage dataset across 38 crop and disease classes:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 98.85% |
| **Precision** | 98.58% |
| **Recall** | 98.42% |
| **F1 Score** | 98.47% |

### Crop Yield Prediction (DecisionTree Regressor)

Evaluated on historical crop yield features:

| Metric | Score |
| :--- | :--- |
| **$R^2$ Score** | 0.974 |
| **Mean Absolute Error (MAE)** | 5,706.45 $hg/ha$ |
| **Root Mean Squared Error (RMSE)** | 13,528.55 $hg/ha$ |

---

## System Architecture & File Layout

```text
.
├── .streamlit/
│   └── config.toml
├── models/
│   ├── DecisionTree_best.pkl
│   ├── efficientnetB0_model_augmented.keras
│   └── preprocessor.pkl
├── results/
│   ├── PlantVillage_results/
│   │   ├── B0/
│   │   ├── B1/
│   │   ├── VGG16/
│   │   ├── inceptionv3/
│   │   ├── mobilenetv2/
│   │   └── resnet50/
│   ├── yield_prediction-results/
│   ├── image.png
│   └── yield.png
├── src/
│   ├── interface/
│   │   ├── app.py
│   │   ├── combined_interface.py
│   │   └── image_interface.py
│   ├── PlantVillage-codes/
│   │   ├── models-training/
│   │   │   ├── ResNet50.py
│   │   │   ├── VGG16.py
│   │   │   ├── efficientnetB0.py
│   │   │   ├── efficientnetB1.py
│   │   │   ├── inception_V3.py
│   │   │   └── mobilenet_V2.py
│   │   ├── data_split.py
│   │   └── test.py
│   └── yield_prediction-codes/
│       ├── split.py
│       ├── train.py
│       └── test.py
├── .gitignore
├── README.md
└── requirements.txt
