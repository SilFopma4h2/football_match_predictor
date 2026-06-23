import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ELO per match
y_elo_home = [8.4, 7.3, 6.8, 6, 6.5]
y_elo_away = [6, 5, 5, 6, 6]

# Goals per match
home_goals = [5, 2, 2, 0, 1]
away_goals = [0, 1, 0, 0, 0]

# Averages, needed for the upcoming prediction
elo_mean_home = np.mean(y_elo_home)
elo_mean_away = np.mean(y_elo_away)
home_goals_mean = np.mean(home_goals)
away_goals_mean = np.mean(away_goals)

# Training data: combine every home match with every away match (5x5 = 25 pairs)
# The label is now based on a form score (ELO and goals combined), not just goals alone.
# This prevents the tree from predicting everything using just 1 of the 2 features.
X_train, y_labels = [], []

for i in range(len(y_elo_home)):
    for j in range(len(y_elo_away)):
        elo_diff = y_elo_home[i] - y_elo_away[j]
        goals_diff = home_goals[i] - away_goals[j]

        form_home = y_elo_home[i] + home_goals[i]
        form_away = y_elo_away[j] + away_goals[j]

        X_train.append([elo_diff, goals_diff])

        if form_home > form_away:
            y_labels.append(1)  # Home wins
        elif form_away > form_home:
            y_labels.append(0)  # Away wins
        else:
            y_labels.append(2)  # Draw

X_train = np.array(X_train)
y_labels = np.array(y_labels)

# Train the tree
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_labels)

# Check which features the tree actually uses
print("Feature importance (ELO / Goals):", tree_model.feature_importances_)

# Prediction for the next match, based on the averages
elo_diff_new = elo_mean_home - elo_mean_away
goals_diff_new = home_goals_mean - away_goals_mean

prediction = tree_model.predict([[elo_diff_new, goals_diff_new]])

label_map = {0: "Away wins", 1: "Home wins", 2: "Draw"}
names_in_correct_order = [label_map[c] for c in tree_model.classes_]

print(f"\nPrediction for Home vs Away: {label_map[prediction[0]]}")

# Plot the tree
plt.figure(figsize=(12, 7))
plot_tree(
    tree_model,
    feature_names=["ELO Difference", "Goals Difference"],
    class_names=names_in_correct_order,
    filled=True,
    rounded=True,
    fontsize=12,
    impurity=False
)
plt.savefig("football_match_predictor/assets/tree.png")
plt.title("Decision Tree: Home vs Away", fontsize=15)
plt.tight_layout()
plt.show()


