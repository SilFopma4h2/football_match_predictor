import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import os
import json


def plot_tree_ml():

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    json_path = os.path.join(BASE_DIR, "assets", "data", "data.json")
    save_path = os.path.join(BASE_DIR, "assets", "tree.png")

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

            elo_diff = y_elo_home[i] - y_elo_away[j]
            goals_diff = home_goals[i] - away_goals[j]
            xg_diff = nl_xg[i] - ma_xg[j]
            shots_diff = nl_shots[i] - ma_shots[j]
            pos_diff = nl_possession[i] - ma_possession[j]
            pass_diff = nl_passes[i] - ma_passes[j]

            X_train.append([elo_diff, goals_diff, xg_diff, shots_diff, pos_diff, pass_diff])

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

    means_home = [
        np.mean(y_elo_home), np.mean(home_goals), np.mean(nl_xg),
        np.mean(nl_shots), np.mean(nl_possession), np.mean(nl_passes)
    ]

    means_away = [
        np.mean(y_elo_away), np.mean(away_goals), np.mean(ma_xg),
        np.mean(ma_shots), np.mean(ma_possession), np.mean(ma_passes)
    ]

    prediction_features = [means_home[k] - means_away[k] for k in range(6)]
    prediction = tree_model.predict([prediction_features])

    label_map = {0: "Away wins", 1: "Home wins", 2: "Draw"}

    print(f"\nFinal Prediction: {label_map[prediction[0]]}")

    plt.figure(figsize=(20, 10))

    feature_names = [
        "ELO Diff", "Goals Diff", "xG Diff",
        "Shots Diff", "Possession Diff", "Passes Diff"
    ]

    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=["Away wins", "Home wins", "Draw"],
        filled=True,
        rounded=True,
        fontsize=8,
        impurity=False
    )

    plt.title("Decision Tree Model")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_tree_ml()