import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import json
import os


class FootballPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size1=32, hidden_size2=16, num_classes=3, dropout_prob=0.2):
        super(FootballPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


def get_winner1():

    torch.manual_seed(42)
    np.random.seed(42)

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    json_path = os.path.join(BASE_DIR, "assets", "data", "data.json")

    with open(json_path, "r") as f:
        data = json.load(f)

    elo_home = data["elo_home"]
    elo_away = data["elo_away"]
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

    means_home = [
        np.mean(elo_home), np.mean(home_goals), np.mean(nl_xg),
        np.mean(nl_shots), np.mean(nl_possession), np.mean(nl_passes)
    ]

    means_away = [
        np.mean(elo_away), np.mean(away_goals), np.mean(ma_xg),
        np.mean(ma_shots), np.mean(ma_possession), np.mean(ma_passes)
    ]

    X_train, y_labels = [], []

    for i in range(len(elo_home)):
        for j in range(len(elo_away)):

            X_train.append([
                elo_home[i] - elo_away[j],
                home_goals[i] - away_goals[j],
                nl_xg[i] - ma_xg[j],
                nl_shots[i] - ma_shots[j],
                nl_possession[i] - ma_possession[j],
                nl_passes[i] - ma_passes[j]
            ])

            score_home = (
                elo_home[i] + home_goals[i] + nl_xg[i] +
                (nl_shots[i] / 10.0) + (nl_possession[i] / 20.0) +
                (nl_passes[i] / 100.0)
            )

            score_away = (
                elo_away[j] + away_goals[j] + ma_xg[j] +
                (ma_shots[j] / 10.0) + (ma_possession[j] / 20.0) +
                (ma_passes[j] / 100.0)
            )

            if score_home > score_away:
                y_labels.append(1)
            elif score_away > score_home:
                y_labels.append(0)
            else:
                y_labels.append(2)

    X_train = np.array(X_train, dtype=np.float32)
    y_labels = np.array(y_labels, dtype=np.int64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_labels)

    model = FootballPredictor()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    prediction_input = [
        means_home[k] - means_away[k] for k in range(6)
    ]

    prediction_input = scaler.transform([prediction_input])
    prediction_input = torch.from_numpy(prediction_input.astype(np.float32))

    model.eval()
    with torch.no_grad():
        output = model(prediction_input)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).item()

    label_map = {0: "Away wins", 1: "Home wins", 2: "Draw"}

    print(f"Away: {probs[0][0]:.2f}")
    print(f"Home: {probs[0][1]:.2f}")
    print(f"Draw: {probs[0][2]:.2f}")

    print("Prediction:", label_map[pred])

    return label_map[pred]