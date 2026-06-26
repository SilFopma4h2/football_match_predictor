import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


# Define the upgraded neural network model
class FootballPredictor(nn.Module):
    # input_size is now 6 to accommodate all features
    def __init__(self, input_size=6, hidden_size1=32, hidden_size2=16, num_classes=3, dropout_prob=0.2):
        super(FootballPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.dropout1(out)
        out = self.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)
        return out


def get_winner1():
    # Fix seeds for consistency
    torch.manual_seed(42)
    np.random.seed(42)

    # --- COMPLETE DATA INTEGRATION ---

    # 1. Original Data (ELO and Goals)
    elo_home = [7.5, 8.4, 7.3, 6.8, 6.0, 6.5]
    elo_away = [7.0, 5.0, 7.0, 6.0, 7.5, 7.5]
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

    # Calculate averages for the upcoming match prediction
    means_home = [
        np.mean(elo_home), np.mean(home_goals), np.mean(nl_xg),
        np.mean(nl_shots), np.mean(nl_possession), np.mean(nl_passes)
    ]
    means_away = [
        np.mean(elo_away), np.mean(away_goals), np.mean(ma_xg),
        np.mean(ma_shots), np.mean(ma_possession), np.mean(ma_passes)
    ]

    # Generate training data combinations
    X_train, y_labels = [], []
    for i in range(len(elo_home)):
        for j in range(len(elo_away)):
            # Features: Differences in ELO, Goals, xG, Shots, Possession, Passes
            diffs = [
                elo_home[i] - elo_away[j],
                home_goals[i] - away_goals[j],
                nl_xg[i] - ma_xg[j],
                nl_shots[i] - ma_shots[j],
                nl_possession[i] - ma_possession[j],
                nl_passes[i] - ma_passes[j]
            ]
            X_train.append(diffs)

            # Holistic performance score for labeling
            score_home = elo_home[i] + home_goals[i] + nl_xg[i] + (nl_shots[i] / 10.0) + (nl_possession[i] / 20.0) + (
                        nl_passes[i] / 100.0)
            score_away = elo_away[j] + away_goals[j] + ma_xg[j] + (ma_shots[j] / 10.0) + (ma_possession[j] / 20.0) + (
                        ma_passes[j] / 100.0)

            if score_home > score_away:
                y_labels.append(1)  # Home wins
            elif score_away > score_home:
                y_labels.append(0)  # Away wins
            else:
                y_labels.append(2)  # Draw

    X_train = np.array(X_train, dtype=np.float32)
    y_labels = np.array(y_labels, dtype=np.int64)

    # Normalize the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X_train_scaled)
    y_tensor = torch.from_numpy(y_labels)

    # Initialize model with 6 inputs and slightly larger hidden layers
    model = FootballPredictor(input_size=6, hidden_size1=32, hidden_size2=16)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 300  # Increased epochs for more features
    print("Training Upgraded Neural Network...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Prepare prediction data (average differences)
    prediction_diffs = [means_home[k] - means_away[k] for k in range(6)]
    new_sample = np.array([prediction_diffs], dtype=np.float32)

    # Apply normalization
    new_sample_scaled = scaler.transform(new_sample)
    new_data = torch.from_numpy(new_sample_scaled)

    # Predict outcome
    model.eval()
    with torch.no_grad():
        output = model(new_data)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()

    # Map labels
    label_map = {0: "Away wins (Morocco)", 1: "Home wins (Netherlands)", 2: "Draw"}
    winner = label_map[predicted_class]

    print(
        f"\nProbabilities: Home: {probabilities[0][1]:.2f}, Away: {probabilities[0][0]:.2f}, Draw: {probabilities[0][2]:.2f}")
    print(f"Prediction for Netherlands vs Morocco: {winner}")
    return winner



