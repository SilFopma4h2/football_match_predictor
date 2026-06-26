import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


def get_winner():
    # --- FULL DATA INTEGRATION ---

    # 1. Original Data (ELO and Goals)
    y_elo_home = [7.5, 8.4, 7.3, 6.8, 6.0, 6.5]
    y_elo_away = [7.0, 5.0, 7.0, 6.0, 7.5, 7.5]
    home_goals = [3, 5, 2, 2, 0, 1]
    away_goals = [4, 1, 1, 1, 4, 5]

    # 2. Advanced Data for Nederland (Netherlands)
    nl_xg = [2.45, 2.07, 1.80, 0.70, 1.40, 1.62]
    nl_possession = [71.6, 62.1, 58.0, 52.0, 61.0, 54.0]
    nl_passes = [601, 521, 520, 480, 610, 506]
    nl_shots = [20, 11, 14, 9, 12, 15]

    # 3. Advanced Data for Marokko (Morocco)
    ma_xg = [2.8, 1.5, 0.9, 1.2, 2.1, 2.4]
    ma_possession = [66.0, 53.0, 44.0, 49.0, 60.0, 62.0]
    ma_passes = [720, 480, 390, 430, 590, 610]
    ma_shots = [20, 12, 8, 10, 16, 18]

    # Averages for the upcoming prediction
    means_home = [
        np.mean(y_elo_home), np.mean(home_goals), np.mean(nl_xg),
        np.mean(nl_shots), np.mean(nl_possession), np.mean(nl_passes)
    ]
    means_away = [
        np.mean(y_elo_away), np.mean(away_goals), np.mean(ma_xg),
        np.mean(ma_shots), np.mean(ma_possession), np.mean(ma_passes)
    ]

    # --- TRAINING DATA PREPARATION ---
    # Features: ELO Diff, Goals Diff, xG Diff, Shots Diff, Possession Diff, Passes Diff
    X_train, y_labels = [], []

    for i in range(len(y_elo_home)):
        for j in range(len(y_elo_away)):
            # Feature calculation
            diffs = [
                y_elo_home[i] - y_elo_away[j],
                home_goals[i] - away_goals[j],
                nl_xg[i] - ma_xg[j],
                nl_shots[i] - ma_shots[j],
                nl_possession[i] - ma_possession[j],
                nl_passes[i] - ma_passes[j]
            ]
            X_train.append(diffs)

            # Label calculation: Using a complete performance score
            score_home = y_elo_home[i] + home_goals[i] + nl_xg[i] + (nl_shots[i] / 10.0) + (nl_possession[i] / 20.0) + (
                        nl_passes[i] / 100.0)
            score_away = y_elo_away[j] + away_goals[j] + ma_xg[j] + (ma_shots[j] / 10.0) + (ma_possession[j] / 20.0) + (
                        ma_passes[j] / 100.0)

            if score_home > score_away:
                y_labels.append(1)  # Home wins
            elif score_away > score_home:
                y_labels.append(0)  # Away wins
            else:
                y_labels.append(2)  # Draw

    X_train = np.array(X_train)
    y_labels = np.array(y_labels)

    # --- MODEL TRAINING ---
    tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_model.fit(X_train, y_labels)

    # Feature Importance Check
    feature_names = ["ELO Diff", "Goals Diff", "xG Diff", "Shots Diff", "Possession Diff", "Passes Diff"]
    print("Complete Feature Importances:")
    for name, importance in zip(feature_names, tree_model.feature_importances_):
        print(f"{name}: {importance:.4f}")

    # --- PREDICTION ---
    prediction_diffs = [means_home[k] - means_away[k] for k in range(6)]
    prediction = tree_model.predict([prediction_diffs])

    label_map = {0: "Away wins", 1: "Home wins", 2: "Draw"}
    final_winner = label_map[prediction[0]]

    print(f"\nFinal Complete Prediction (Home vs Away): {final_winner}")

    return final_winner



