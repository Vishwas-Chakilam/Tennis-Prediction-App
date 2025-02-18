import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# ------------------------------
# DATA & MODEL FUNCTIONS
# ------------------------------

@st.cache_data
def load_data():
    # Use a raw string to avoid escape sequence issues
    data = pd.read_csv(r'E:\Machine Learning\playtennis\play_tennis.csv')
    return data

def preprocess_data(data):
    label_encoders = {}
    for column in ['outlook', 'temp', 'humidity', 'wind', 'play']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders

def train_model(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

# ------------------------------
# PAGE FUNCTIONS
# ------------------------------

def home_page():
    st.title("🎾 Welcome to the Play Tennis Prediction App! 🚀")

    st.write("""
    Curious if today's weather is perfect for a tennis match?  
    This app **predicts** whether you should play tennis based on the **weather conditions** using a **Decision Tree Classifier**.  
    Simply head over to the **Predict** page, enter the weather details, and get instant insights!  
    """)

    st.image(
        "https://images.pexels.com/photos/2339377/pexels-photo-2339377.jpeg?auto=compress&cs=tinysrgb&w=600", 
        caption="Let's play tennis!", 
        use_container_width=True
    )

    st.subheader("📌 Introduction")
    st.write("""
    Tennis is a game that requires the right weather conditions for the best experience.  
    This application leverages **Machine Learning (Decision Tree Algorithm)** to analyze weather data and predict if it's a **good day** to play tennis.  
    Whether you're a casual player or a pro, this app can help you plan your game better! 🎾🔥  
    """)

    st.markdown("---")

    st.subheader("🚀 How It Works?")
    st.write("""
    - Navigate to the **Predict** page.  
    - Enter the weather conditions (Outlook, Temperature, Humidity, Wind).  
    - Click **Predict** and get a recommendation instantly.  
    - Simple, fast, and accurate!  
    """)

    st.success("🏆 Ready to check if you can play? Head over to the **Predict** page now!")



# Predict Page
def predict_page():
    st.title("🎾 Play Tennis Prediction")
    st.write("Provide the weather conditions to predict whether it's a good day for tennis! 🌦️")

    # Load and preprocess the data
    data = load_data()
    data, label_encoders = preprocess_data(data)

    # Model training
    X = data.drop(['play', 'day'], axis=1)
    y = data['play']
    model = train_model(X, y)

    # User input form
    with st.form("predict_form", clear_on_submit=True):
        st.subheader("🌤️ Weather Conditions")
        
        col1, col2 = st.columns(2)
        with col1:
            outlook = st.selectbox("🌞 Outlook", ["Sunny", "Overcast", "Rain"], key="outlook")
            temp = st.selectbox("🌡️ Temperature", ["Hot", "Mild", "Cool"], key="temp")
        with col2:
            humidity = st.selectbox("💧 Humidity", ["High", "Normal"], key="humidity")
            wind = st.selectbox("💨 Wind", ["Weak", "Strong"], key="wind")

        submit_button = st.form_submit_button("⚡ Predict", use_container_width=True)

    # Make prediction
    if submit_button:
        new_data = pd.DataFrame({
            'outlook': [outlook],
            'temp': [temp],
            'humidity': [humidity],
            'wind': [wind]
        })

        # Encode user input
        for column in new_data.columns:
            new_data[column] = label_encoders[column].transform(new_data[column])

        # Model prediction
        prediction = model.predict(new_data)
        result = label_encoders['play'].inverse_transform(prediction)[0]

        # Display Result
        st.subheader("🧐 Prediction Result")
        if result == "Yes":
            st.success("🎾 Yes! You should play tennis today. Have fun! 🏸")
        else:
            st.error("🚫 No! It's not a great day for tennis. Maybe try another time! ☁️")

    # Footer



def about_page():
    st.title("ℹ️ About This Project")

    st.write("""
    This project is built as a **Machine Learning-based prediction system** that helps tennis players determine if they can play based on current weather conditions.  
    It is powered by **Decision Tree Classifier** and developed using **Streamlit** for an interactive UI.
    """)

    st.subheader("📊 How the Model Works?")
    st.write("""
    - The dataset contains **historical weather conditions** and whether tennis was played.  
    - It includes attributes like **Outlook, Temperature, Humidity, and Wind**.  
    - A **Decision Tree Model** is trained on this dataset to learn patterns.  
    - The model makes predictions based on new weather inputs! ✅  
    """)

    st.markdown("---")

    st.subheader("👨‍💻 About the Developer")
    st.write("""
    This app is developed by **[Vishwas Chakilam](https://github.com/vishwas-chakilam)**, a passionate **Machine Learning & Data Science Enthusiast**.  
    Explore my other projects on GitHub and feel free to contribute!  
    """)

    # GitHub profile card
    st.markdown("""
    <div align="center">
        <a href="https://github.com/vishwas-chakilam" target="_blank">
            <img src="https://ghchart.rshah.org/00b4d8/vishwas-chakilam" alt="GitHub Heatmap" width="600px"/>
        </a>
    </div>
""", unsafe_allow_html=True)


    st.markdown("---")

    st.subheader("📬 Get in Touch!")
    st.write("""
    - 🌎 **GitHub**: [github.com/vishwas-chakilam](https://github.com/vishwas-chakilam)  
    - ✉️ **Email**: work.vishwas1@gmail.com 
    - 📢 Always open for collaborations & discussions on AI & ML! 🚀  
    """)

    st.success("🔗 Check out my GitHub profile for more cool projects!")


# ------------------------------
# MAIN APP
# ------------------------------

def main():
    # Custom CSS for a modern look
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
            body {
                font-family: 'Roboto', sans-serif;
                background: #f0f2f6;
            }
            .reportview-container .main {
                background: linear-gradient(145deg, #e0eafc, #cfdef3);
                padding: 50px;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .sidebar .sidebar-content {
                background-color: #2c2f33;
                color: #ffffff;
                padding: 20px;
                border-radius: 15px;
            }
            h1, h2, h3 {
                color: #333;
                font-weight: 700;
            }
            .stButton>button {
                background-color: #0077b6;
                color: #fff;
                border: none;
                border-radius: 10px;
                padding: 15px 30px;
                font-size: 16px;
                transition: background-color 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #0096c7;
            }
            footer {
                text-align: center;
                font-size: 12px;
                margin-top: 50px;
                color: #666;
            }
            footer a {
                color: #0077b6;
                text-decoration: none;
            }
            hr {
                border: 0;
                height: 1px;
                background: #ccc;
                margin-top: 30px;
                margin-bottom: 30px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Navbar simulation using sidebar radio buttons
    nav = st.sidebar.radio("Menu", ["Home", "Predict", "About"])

    if nav == "Home":
        home_page()
    elif nav == "Predict":
        predict_page()
    elif nav == "About":
        about_page()

    # Global Footer
    st.markdown("""
        <hr>
        <footer>
            Developed with ❤️ by <a href="https://github.com/vishwas-chakilam" target="_blank">Vishwas Chakilam</a> | Open-source Project 💻
        </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()