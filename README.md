# Iris Flower Classifier

A Streamlit web app that predicts an iris flower species from four measurements using a trained K-nearest neighbors model.

## Features

- Interactive sliders for sepal and petal measurements
- Predictions for Setosa, Versicolor, and Virginica
- Colorful dashboard with measurement summaries and species information
- Model trained from the Iris dataset

## Requirements

- Python 3.9 or newer
- The packages listed in `requirements.txt`

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, use `.venv\\Scripts\\activate` instead.

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the app

From the project directory, run:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Project files

- `app.py` - Streamlit user interface and prediction logic
- `iris_model.pkl` - Saved trained K-nearest neighbors model
- `Iris.csv` - Iris dataset
- `model.ipynb` - Notebook used for data exploration and model training
- `requirements.txt` - Python dependencies

## How it works

The app accepts these four measurements in centimeters:

- Sepal length
- Sepal width
- Petal length
- Petal width

These values are passed to the saved model, which predicts the most likely iris species.

## Retraining the model

To retrain or change the model, open `model.ipynb`. After training, save the model as `iris_model.pkl` in the project directory so that `app.py` can load it.
