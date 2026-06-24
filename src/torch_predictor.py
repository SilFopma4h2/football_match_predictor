import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Define the neural network model
class FootballPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size1=16, hidden_size2=8, num_classes=3, dropout_prob=0.2):
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

def get_winner():
    # Zet de seeds vast voor consistentie over verschillende runs
    torch.manual_seed(42)
    np.random.seed(42)

    # Data: ELO and goals per match
    elo_home = [8.4, 7.3, 6.8, 6, 6.5]
    elo_away = [6, 5, 5, 6, 6]
    home_goals = [5, 2, 2, 0, 1]
    away_goals = [0, 1, 0, 0, 0]

    # Calculate averages for the upcoming match prediction
    mean_elo_home = np.mean(elo_home)
    mean_elo_away = np.mean(elo_away)
    mean_home_goals = np.mean(home_goals)
    mean_away_goals = np.mean(away_goals)

    # Generate training data: combine each home match with each away match (5 x 5 = 25 combinations)
    X_train, y_labels = [], []
    for i in range(len(elo_home)):
        for j in range(len(elo_away)):
            elo_diff = elo_home[i] - elo_away[j]
            goals_diff = home_goals[i] - away_goals[j]
            form_home = elo_home[i] + home_goals[i]
            form_away = elo_away[j] + away_goals[j]

            X_train.append([elo_diff, goals_diff])
            if form_home > form_away:
                y_labels.append(1)  # Home wins
            elif form_away > form_home:
                y_labels.append(0)  # Away wins
            else:
                y_labels.append(2)  # Draw

    X_train = np.array(X_train, dtype=np.float32)
    y_labels = np.array(y_labels, dtype=np.int64)

    # Normalize the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Convert the data to PyTorch tensors
    X_tensor = torch.from_numpy(X_train_scaled)
    y_tensor = torch.from_numpy(y_labels)

    # Initialize model, loss function, and optimizer
    model = FootballPredictor()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Prepare new data for prediction (using average differences)
    elo_diff_new = mean_elo_home - mean_elo_away
    goals_diff_new = mean_home_goals - mean_away_goals
    new_sample = np.array([[elo_diff_new, goals_diff_new]], dtype=np.float32)

    # Apply the same normalization to the new data
    new_sample_scaled = scaler.transform(new_sample)
    new_data = torch.from_numpy(new_sample_scaled)

    # Predict the outcome
    model.eval()
    with torch.no_grad():
        output = model(new_data)
        predicted_class = torch.argmax(output, dim=1).item()

    # Map the prediction to a label
    label_map = {0: "Away wins", 1: "Home wins", 2: "Draw"}
    winner = label_map[predicted_class]
    print(f"\nPrediction for Home vs Away: {winner}")
    return winner

# Example function to use the winner prediction elsewhere
def use_winner():
    winner = get_winner()
    print(f"The predicted winner is: {winner}")
    return winner

if __name__ == "__main__":
    use_winner()