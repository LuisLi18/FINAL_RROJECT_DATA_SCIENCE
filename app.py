import numpy as np
from flask import Flask, request, render_template
import pickle

model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route('/predict', methods=["POST"])
def predict():
    # Las características se pasan como parámetros en la URL
    # Ejemplo de URL: /predict?horsepower=100&curbweight=2000&enginesize=150
    # horsepower = float(request.args.get('horsepower'))
    # curbweight = float(request.args.get('curbweight'))
    # enginesize = float(request.args.get('enginesize'))
    #
    #
    # # Realiza una predicción con el modelo
    # predicted_price = model.predict([[horsepower, curbweight, enginesize]])

    # return render_template("index.html", prediction_text=f'Precio estimado del automóvil: {predicted_price[0]} dólares')
    # return render_template("index.html", prediction_text=f'Precio estimado del automóvil: dólares')

    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    predicted_price = model.predict(features)
    return render_template("index.html", prediction_text=f'Precio estimado del automóvil: {predicted_price[0]} dólares')

@app.route('/')
def hello():
    return render_template("index.html")


if __name__ == '__main__':
    app.run()
