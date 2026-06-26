import json

import matplotlib.pyplot as plt


with open("data.json", "r") as f:
    data = json.load(f)

#Data plotten uit de JSON
plt.plot(data["elo_home"], label="Home ELO")
plt.plot(data["elo_away"], label="Away ELO")
plt.plot(data["home_goals"], label="Home Goals")
plt.plot(data["away_goals"], label="Away Goals")

plt.title("ELO per match")
plt.xlabel("Match")
plt.ylabel("ELO")


plt.show()
