from flask import Flask,render_template, request
import pandas as pd
from flask import request

car = pd.read_csv("cleaned_car_data.csv")


app = Flask(__name__)


@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(),reverse=True)
    fuel_type = car['fuel_type'].unique()

    return render_template('index.html' , companies = companies, car_models = car_models, years = year, fuel_types = fuel_type)
def load_car_model(company):
    car_models = car[car['company'] == company]['name'].unique()
    return car_models
@app.route('/get_car_models', methods=['POST'])
def get_car_models():
    company = request.form['company']
    car_models = load_car_model(company)  # replace with your function
    return render_template('car_model_options.html', car_models=car_models)


if __name__ == '__main__':
    app.run(debug= True)

