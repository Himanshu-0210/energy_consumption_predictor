# ⚡ Energy Consumption Predictor

A Machine Learning-powered web application built with **Streamlit** to predict household energy consumption based on environmental and usage parameters such as temperature, humidity, wind speed, appliance usage, and date inputs.

## 🚀 Live Demo


```text
https://energyconsumptionpredictor-itmu.streamlit.app/
```

---

## 📌 Project Overview

This project helps estimate future energy consumption using historical patterns and user-provided inputs. It combines **data preprocessing**, **feature engineering**, **machine learning modeling**, and an interactive **Streamlit frontend**.

Users can enter relevant details and instantly receive a predicted energy usage value.

---

## 🧠 Features

* Predict energy consumption in real time
* User-friendly Streamlit interface
* Input fields for date and weather conditions
* Appliance usage as a predictive feature
* Fast and lightweight deployment
* ML model integrated with frontend

---

## 📊 Input Parameters

The app accepts inputs such as:

* Date
* Temperature
* Humidity
* Wind Speed
* Number of Appliances in Use

Based on these inputs, the model predicts estimated energy consumption.

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend / ML

* Python
* Scikit-learn
* Pandas
* NumPy
* Joblib / Pickle

### Deployment

* Streamlit Community Cloud

---

## 📁 Project Structure

```text
energy_consumption_predictor/
│── app.py
│── model.pkl
│── requirements.txt
│── README.md
│── dataset.csv
```

---

## ⚙️ Installation & Run Locally

Clone the repository:

```bash
git clone https://github.com/Himanshu-0210/energy_consumption_predictor.git
cd energy_consumption_predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🤖 Machine Learning Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Train-Test Split
5. Model Training
6. Hyperparameter Tuning
7. Evaluation using MAE / RMSE / R² Score
8. Deployment with Streamlit

---

## 📈 Future Improvements

* Daily / Weekly forecasting
* Graphical trend analysis
* Energy-saving recommendations
* Advanced ensemble models
* Better UI/UX enhancements

---

## 👨‍💻 Author

**Himanshu Upadhyay**
GitHub: https://github.com/Himanshu-0210

---

## 📜 License

This project is open-source and available under the MIT License.
