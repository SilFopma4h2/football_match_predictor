# Football Match Predictor

This project contains two methods for predicting the outcome of a football match – a Decision Tree Classifier and a PyTorch-based neural network – both using previous match data based on ELO ratings and goal differences. The goal is to compare the predictions from both approaches and ensure consistency between them.

---

## Overview

1. **Input Data** For every match, separate values are recorded for:
   - ELO ratings (for the home and away teams)
   - Goals scored (for the home and away teams)

2. **Generating Training Data** Both methods generate training data by combining every home match with every away match (5 × 5 = 25 pairs). For each pair, the following values are computed:
   - `elo_diff`: Difference in ELO ratings.
   - `goals_diff`: Difference in goals scored.
   
   In addition, a "form score" is calculated by adding the ELO rating and goal count. This combined score determines the outcome:
   - Home win
   - Away win
   - Draw

3. **Models**
   - **Decision Tree (`ml_predictor.py`):** Uses scikit-learn's `DecisionTreeClassifier` with a maximum depth of 3. It outputs the feature importances and performs a prediction based on the average differences of ELO and goals.
   
   - **Neural Network (`torch_predictor.py`):** Implements a simple feed-forward neural network with two hidden layers using PyTorch. It normalizes the input data using `StandardScaler`, trains on the 25 training samples, and predicts the outcome based on average differences.

4. **Comparing Predictions (`main_test.py`)** A dedicated module imports both predictor functions and compares the resulting predictions to ensure both methodologies reach the same conclusion.

---

## Installation

Ensure you have Python 3.12.8 installed. Then install the required packages by executing:

```bash
pip install -r requirements.txt
```

The project depends on, but is not limited to, the following libraries:

- numpy
- matplotlib
- scikit-learn
- torch
- and any additional dependencies as specified in `requirements.txt`

---

## Usage

The project contains several Python modules:

- **`ml.py`**: Executes prediction using the Decision Tree classifier. It prints the feature importance and the match outcome based on the computed average values.
- **`te.py`**: Executes prediction using the neural network model developed with PyTorch. It details the training progress (loss through epochs) and prints the final prediction.
- **`oe.py`**: This module imports both the Decision Tree and the neural network predictors to compare their predictions. Run this module to verify that both models agree on the outcome:

```bash
python main_test.py
```

During execution, the program outputs:
- Feature importances (for the Decision Tree).
- Training status (loss values during training epochs for the neural network).
- The predicted match outcome (Home wins / Away wins / Draw).

---

## Customizing Data

You can replace the built-in data by modifying the respective lists in both `ml.py` and `te.py`. Make sure that:

- The lists for ELO ratings and goals for both home and away matches have the same length.
- The data is consistent across modules to ensure valid comparisons between the two methods.

---

## License

This project is licensed under the MIT License. See the LICENSE file for further details.