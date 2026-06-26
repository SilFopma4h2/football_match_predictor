from flask import Flask, render_template, request
import importlib

app = Flask(__name__)

HOME_TEAM = "Marokko"
AWAY_TEAM = "Nederland"


def resolve_predict_function(module):
    """
    Zoek een bruikbare predict-functie in module.
    """
    candidates = [
        "predict_winner",
        "predict_match",
        "predict",
        "get_prediction",
        "run_prediction",
    ]
    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def call_model(module_name: str, model_label: str) -> tuple[str | None, str | None]:
    """
    Return: (winner, error)
    """
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        return None, f"Kon module '{module_name}' niet laden: {e}"

    fn = resolve_predict_function(module)
    if not fn:
        return None, (
            f"Geen predict-functie gevonden in {module_name}. "
            "Gebruik bijv. predict_winner(home_team, away_team)."
        )

    # Probeer meerdere call-signatures
    attempts = [
        lambda: fn(HOME_TEAM, AWAY_TEAM),                    # fn(home, away)
        lambda: fn({"home_team": HOME_TEAM, "away_team": AWAY_TEAM}),  # fn(dict)
        lambda: fn(home_team=HOME_TEAM, away_team=AWAY_TEAM),          # fn(kwargs)
    ]

    last_error = None
    for attempt in attempts:
        try:
            result = attempt()
            # normaliseer output
            if isinstance(result, dict):
                winner = (
                        result.get("winner")
                        or result.get("prediction")
                        or result.get("predicted_winner")
                )
                if winner:
                    return str(winner), None
                return str(result), None
            if result is None:
                continue
            return str(result), None
        except Exception as e:
            last_error = e

    return None, f"Model gaf geen bruikbare output. Laatste fout: {last_error}"


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    selected_model = "ml"

    if request.method == "POST":
        selected_model = request.form.get("model_choice", "ml").strip().lower()

        if selected_model == "torch":
            winner, err = call_model("src.torch_predictor", "PyTorch Model")
            model_label = "PyTorch Model"
        else:
            winner, err = call_model("src.ml_predictor", "ML Model (Decision Tree)")
            model_label = "ML Model (Decision Tree)"

        if err:
            error = err
        else:
            prediction = {
                "winner": winner,
                "model_label": model_label,
                "home_team": HOME_TEAM,
                "away_team": AWAY_TEAM,
            }

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        selected_model=selected_model,
        home_team=HOME_TEAM,
        away_team=AWAY_TEAM,
    )


if __name__ == "__main__":
    app.run(debug=True)