from datetime import datetime, date
from flask import Flask, request, render_template
import config
import os
import pandas as pd
import numpy as np
import json
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import io
from io import BytesIO
import base64
import tensorflow as tf

app = Flask(__name__)

@app.route('/')
def predict():
    return render_template('predict.html')


@app.route('/explore')
def explore():
    g1, g2, g3, g4, g5 = show_explore_page()
    return render_template('explore.html', grafico1 = g1, grafico2 = g2, grafico3=g3, grafico4 = g4, grafico5 = g5)

##################################################################################################################


# Predict Page

def load_model():
    model = tf.keras.models.load_model('tensorflow_model.h5')
    return model


def calculate_age(age):
    if age < 18:
        age = "Under 18 years old"
    elif age < 25:
        age = "18-24 years old"
    elif age < 35:
        age = "25-34 years old"
    elif age < 45:
        age = "35-44 years old"
    elif age < 55:
        age = "45-54 years old"
    elif age < 65:
        age = "55-64 years old"
    else:
        age = "65 years or older"
    return age


@app.route('/prediction', methods=['POST'])
def prediction():
    # Create a dataframe from the HTML form
    input = request.get_json()

    country = input['country']
    age = int(input['age'])
    gender = input['gender']
    education = input['education']
    experience = int(input['experience'])

    model = load_model()

    if experience >= age:
        res=("ERROR: Experience is greater than age")
        return res

    elif age - experience < 15:
        res=("ERROR: The minimum age for access to work cannot be less than 15 years old")
        return res

    else:
        age = calculate_age(age)

        data = ['18-24 years old', '25-34 years old', '35-44 years old',
                '45-54 years old', '55-64 years old', '65 years or older',
                'Under 18 years old',  'Man', 'Woman', 'Other',
                'Australia', 'Brazil', 
                'Canada', 'France', 'Germany', 'India',
                'Italy', 'Netherlands', 'Poland',
                'Russian Federation', 'Spain', 'Sweden',
                'Switzerland', 'United Kingdom of Great Britain and Northern Ireland',
                'United States of America', 'Bachelor’s degree',
                'Less than a Bachelors', 'Master’s degree', 'Post grad', experience]
        
        X = pd.DataFrame([data], columns=['Age_18-24 years old', 'Age_25-34 years old', 'Age_35-44 years old',
                'Age_45-54 years old', 'Age_55-64 years old', 'Age_65 years or older',
                'Age_Under 18 years old', 'Gender_Man', 'Gender_Woman', 'Gender_Other',
                'Country_Australia', 'Country_Brazil', 
                'Country_Canada', 'Country_France', 'Country_Germany', 'Country_India',
                'Country_Italy', 'Country_Netherlands', 'Country_Poland',
                'Country_Russian Federation', 'Country_Spain', 'Country_Sweden',
                'Country_Switzerland', 'Country_United Kingdom of Great Britain and Northern Ireland',
                'Country_United States of America', 'EdLevel_Bachelor’s degree',
                'EdLevel_Less than a Bachelors', 'EdLevel_Master’s degree', 'EdLevel_Post grad', 'YearsCodePro'])
        
        # Ottieni tutte le colonne tranne l'ultima (usando -1 per l'indice dell'ultima colonna)
        colonne_da_esaminare = X.columns[:-1]

        # Itera attraverso le colonne da esaminare
        for column in colonne_da_esaminare:
            if X[column][0] == country or X[column][0] == age or X[column][0] == education or X[column][0] == gender:
                X.at[0, column] = 1
            else:
                X.at[0, column] = 0

        # Convert all integer columns to float
        X = X.astype(float)

        salary = model.predict(X)
        salary = round(salary[0][0], 2)
        res = ("The estimated salary is $" + str(salary))
        return res
    
##################################################################################################################


# Explore Page

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)


def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'


def clean_gender(x):
    if x ==  'Man':
        return 'Man'
    if x == 'Woman':
        return 'Woman'
    else:
        return 'Other'


def load_data():
    df = pd.read_csv("survey_results_public.csv")
    df = df[["Country", "Age", "Gender", "EdLevel", "YearsCodePro", "Employment", "ConvertedCompYearly"]]
    df = df[df["ConvertedCompYearly"].notnull()]
    df = df[df["Age"].notnull()]
    df = df[df["Age"]!="Prefer not to say"]
    df = df[df["Gender"].notnull()]
    df = df[df["Gender"]!="Prefer not to say"]
    df = df.dropna()
    df = df[df["Employment"] == "Employed, full-time"]
    df = df.drop("Employment", axis=1)

    country_map = shorten_categories(df.Country.value_counts(), 400)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["ConvertedCompYearly"] <= 250000]
    df = df[df["ConvertedCompYearly"] >= 10000]
    df = df[df["Country"] != "Other"]

    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)
    df["Gender"] = df["Gender"].apply(clean_gender)
    df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
    return df


def show_explore_page():

    df = load_data()

    # Number of Data from different countries

    df["Country"] = df["Country"].replace("United Kingdom of Great Britain and Northern Ireland", "United Kingdom")
    data = df["Country"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")  # Cambia il formato in 'jpg' se vuoi un'immagine JPG
    plt.close()
    buffer.seek(0)

    grafico1 = base64.b64encode(buffer.read()).decode()


    # Mean Salary Based On Country

    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    plt.figure(figsize=(10, 8))  # Imposta le dimensioni del grafico (larghezza, altezza)
    data.plot(kind='bar')
    plt.xlabel("Country")
    plt.ylabel("Salary")

    plt.subplots_adjust(bottom=0.3)  # Aggiungi spazio in basso per l'etichetta dell'asse x
    plt.xticks(rotation=45, ha="right")  # Ruota e allinea le etichette sull'asse x
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    bar_chart_data = base64.b64encode(buffer.read()).decode()
    grafico2 = bar_chart_data


    # Mean Salary Based On Country and Gender

    # Filtra il DataFrame per includere solo le righe con "Gender" uguale a "Man" (uomo)
    men_salaries = df[df['Gender'] == 'Man']
    men_salaries_by_country = men_salaries.groupby("Country")["Salary"].mean()
    std_men = men_salaries.groupby("Country")["Salary"].std()

    # Filtra il DataFrame per includere solo le righe con "Gender" uguale a "Woman" (donna)
    women_salaries = df[df['Gender'] == 'Woman']
    women_salaries_by_country = women_salaries.groupby("Country")["Salary"].mean()
    std_women = women_salaries.groupby("Country")["Salary"].std()

    # Crea un array di indici per il grafico
    index = np.arange(len(men_salaries_by_country))

    # Imposta le dimensioni del grafico
    plt.figure(figsize=(10, 8))

    # Crea il grafico a barre
    bar_width = 0.35
    plt.bar(index, men_salaries_by_country, width=bar_width, alpha=0.6, color='b', yerr=std_men, error_kw={'ecolor': '0.3'}, label='Men')
    plt.bar(index + bar_width, women_salaries_by_country, width=bar_width, alpha=0.6, color='r', yerr=std_women, error_kw={'ecolor': '0.3'}, label='Women')

    # Imposta etichette e legenda
    plt.xlabel('Country')
    plt.ylabel('Average Salary')
    plt.xticks(index + bar_width / 2, men_salaries_by_country.index, rotation=45, ha='right')
    plt.legend()

    # Salva il grafico come un'immagine PNG
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    bar_chart_data = base64.b64encode(buffer.read()).decode()
    grafico5 = bar_chart_data


    # Mean Salary Based On Age

    data = df.groupby(["Age"])["Salary"].mean()
    data = data.sort_index(ascending=True)

    # Riordina l'indice per mettere "Under 18 years old" come primo
    order = ["Under 18 years old", "18-24 years old", "25-34 years old", "35-44 years old", "45-54 years old", "55-64 years old", "65 years or older"]
    data= data.reindex(order)

    plt.figure(figsize=(10, 8))  # Imposta le dimensioni del grafico (larghezza, altezza)
    plt.plot(data.index, data.values, marker='o')  # Crea un grafico a linee con marcatori
    plt.xlabel("Age")
    plt.ylabel("Mean Salary")
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    line_chart_data1 = base64.b64encode(buffer.read()).decode()
    grafico3 = line_chart_data1


    #Mean Salary Based On Experience

    data = df.groupby(["YearsCodePro"])["Salary"].mean()
    data = data.sort_index(ascending=True)
    plt.figure(figsize=(10, 8))  # Imposta le dimensioni del grafico (larghezza, altezza)
    plt.subplots_adjust(bottom=0.2)  # Aggiungi spazio in basso per l'etichetta dell'asse x
    plt.plot(data.index, data.values, marker='o')  # Crea un grafico a linee con marcatori
    plt.xlabel("Years of Professional Coding Experience")
    plt.ylabel("Mean Salary")
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    line_chart_data2 = base64.b64encode(buffer.read()).decode()
    grafico4 = line_chart_data2

    return grafico1, grafico2, grafico3, grafico4, grafico5




##################################################################################################################

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)