# AI Blood Bank Inventory Management System

## Overview

This project is an **AI Blood Bank Inventory Management System** that utilizes **Machine Learning** and **Reinforcement Learning** for blood demand forecasting and inventory optimization. The system includes a **Flask Web Application**, a **Chatbot for assistance**, and a **data-driven approach** for efficient blood usage management.

## Features

- **Data Preprocessing:** Cleans and normalizes blood usage data.
- **Machine Learning Model:** LSTM-based demand forecasting.
- **Reinforcement Learning:** PPO-based inventory optimization.
- **Flask Web Application:** User-friendly interface.
- **Chatbot Integration:** Provides assistance on loan-related queries.

---

## Project Structure

```
├── Datasets
│   ├── blood_usage_data.csv             # Raw dataset
│   ├── processed_blood_data.csv         # Preprocessed dataset
│
├── Trained_Model
│   ├── blood_demand_lstm.h5             # Trained LSTM model
│   ├── inventory_optimization_rl        # Trained RL model
│
├── static
│   ├── style.css                        # Styles for UI
│
├── templates
│   ├── index.html                       # Main web page
│   ├── chatbot.html                      # Chatbot UI
│
├── data_preprocessing.py                 # Handles dataset cleaning and preprocessing
├── model_training.py                      # Trains ML and RL models
├── app.py                                 # Flask web application
├── README.md                              # Project documentation
```

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/blood-bank-management.git
cd blood-bank-management
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Data Preprocessing

```bash
python data_preprocessing.py
```

### 4. Train the Models

```bash
python model_training.py
```

### 5. Run the Flask App

```bash
python app.py
```

Access the application at: `http://127.0.0.1:5000/`

---

## Technologies Used

- **Python** (pandas, scikit-learn, TensorFlow, gym, stable-baselines3)
- **Flask** (Backend API & Web Server)
- **HTML/CSS/JavaScript** (Frontend UI)
- **GSAP** (Animations for UI/UX)
- **Chatbot** (Loan assistant system)

---

## Future Enhancements

- Integrate real-time blood demand prediction
- Implement authentication for users
- Deploy the project to a cloud platform

---

## Contributor

- **Saai Krishna** - Developer

---

## License

This project is licensed under the MIT License.
