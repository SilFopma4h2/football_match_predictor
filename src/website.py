from flask import Flask, render_template, request

# Importeer de get_winner functie uit je ml_predictor bestand!
from ml_predictor import get_winner

# Let op: gebruik hier eventueel template_folder='../templates' als je map
# nog in de hoofdmap staat in plaats van in src.
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    voorspelling = None

    if request.method == 'POST':
        # Haal de namen van de teams op uit het formulier
        thuis_team = request.form.get('thuis_team')
        uit_team = request.form.get('uit_team')

        # Roep jouw Machine Learning model aan!
        uitkomst = get_winner()

        # Omdat jouw model "Home wins", "Away wins" of "Draw" teruggeeft,
        # vertalen we dat hier naar een mooie Nederlandse zin met de teamnamen erin:
        if uitkomst == "Home wins":
            voorspelling = f"Het model voorspelt dat de thuisploeg ({thuis_team}) gaat winnen!"
        elif uitkomst == "Away wins":
            voorspelling = f"Het model voorspelt dat de uitploeg ({uit_team}) met de winst vandoor gaat!"
        else:
            voorspelling = f"Het model voorspelt een gelijkspel tussen {thuis_team} en {uit_team}!"

    return render_template('index.html', voorspelling=voorspelling)

if __name__ == '__main__':
    app.run(debug=True)