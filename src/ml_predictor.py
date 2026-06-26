import numpy as np
import json
import os
from sklearn.tree import DecisionTreeClassifier


def get_winner():

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    json_path = os.path.join(BASE_DIR, "assets", "data", "data.json")

    with open(json_path, "r") as f:
        data = json.load(f)

    y_elo_home = data["elo_home"]
    y_elo_away = data["elo_away"]
    home_goals = data["home_goals"]
    away_goals = data["away_goals"]

    nl_xg = data["nl_xg"]
    nl_possession = data["nl_possession"]
    nl_passes = data["nl_passes"]
    nl_shots = data["nl_shots"]

    ma_xg = data["ma_xg"]
    ma_possession = data["ma_possession"]
    ma_passes = data["ma_passes"]
    ma_shots = data["ma_shots"]

    X_train, y_labels = [], []

    for i in range(len(y_elo_home)):
        for j in range(len(y_elo_away)):

            X_train.append([
                y_elo_home[i] - y_elo_away[j],
                home_goals[i] - away_goals[j],
                nl_xg[i] - ma_xg[j],
                nl_shots[i] - ma_shots[j],
                nl_possession[i] - ma_possession[j],
                nl_passes[i] - ma_passes[j]
            ])

            score_home = (
                y_elo_home[i] + home_goals[i] + nl_xg[i] +
                (nl_shots[i] / 10.0) + (nl_possession[i] / 20.0) +
                (nl_passes[i] / 100.0)
            )

            score_away = (
                y_elo_away[j] + away_goals[j] + ma_xg[j] +
                (ma_shots[j] / 10.0) + (ma_possession[j] / 20.0) +
                (ma_passes[j] / 100.0)
            )

            if score_home > score_away:
                y_labels.append(1)
            elif score_away > score_home:
                y_labels.append(0)
            else:
                y_labels.append(2)

    X_train = np.array(X_train)
    y_labels = np.array(y_labels)

    tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_model.fit(X_train, y_labels)

    feature_names = [
        "ELO Diff", "Goals Diff", "xG Diff",
        "Shots Diff", "Possession Diff", "Passes Diff"
    ]

    print("Feature Importances:")
    for name, imp in zip(feature_names, tree_model.feature_importances_):
        print(f"{name}: {imp:.4f}")

    # ---- FIX: correct prediction input ----
    prediction_input = [
        np.mean(y_elo_home) - np.mean(y_elo_away),
        np.mean(home_goals) - np.mean(away_goals),
        np.mean(nl_xg) - np.mean(ma_xg),
        np.mean(nl_shots) - np.mean(ma_shots),
        np.mean(nl_possession) - np.mean(ma_possession),
        np.mean(nl_passes) - np.mean(ma_passes)
    ]

    prediction = tree_model.predict([prediction_input])

    label_map = {0: "Away wins", 1: "Home wins", 2: "Draw"}

    final_winner = label_map[prediction[0]]

    print(f"\nFinal Prediction: {final_winner}")

    return final_winner