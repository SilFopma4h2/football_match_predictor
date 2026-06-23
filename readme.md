# Football Match Predictor (Decision Tree)

A simple Python script that uses a **Decision Tree Classifier** to predict the outcome of a football match (Home win / Away win / Draw), based on **ELO rating** and **goal difference** from previous matches.

## How it works

1. **Input data**
   For each match, an ELO score and the number of goals scored are tracked separately for the home team and the away team.

2. **Generating training data**
   Every home match is combined with every away match (5 × 5 = 25 combinations). For each pair, the following is calculated:
   - `elo_diff`: difference in ELO rating
   - `goals_diff`: difference in goals scored

   The label (Home win / Away win / Draw) is determined using a **"form score"**, which combines ELO and goals. This prevents the tree from learning based on only one of the two features.

3. **Training the model**
   A `DecisionTreeClassifier` (max depth 3) is trained on the 25 generated data points.

4. **Prediction**
   For the upcoming match, the average ELO and goal differences of both teams are used to make a prediction.

5. **Visualization**
   The trained tree is plotted using `matplotlib`, including feature names and class names.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python script.py
```

The script prints:
- The feature importance (how heavily ELO and goal difference weigh in the tree)
- The prediction for the upcoming match
- A visual representation of the decision tree

## Customizing

You can easily plug in your own data by editing the lists `y_elo_home`, `y_elo_away`, `home_goals` and `away_goals`. Note: all four lists must have the same length (one value per match).

## Requirements

See `requirements.txt`. The project uses:
- `numpy`
- `matplotlib`
- `scikit-learn`

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.