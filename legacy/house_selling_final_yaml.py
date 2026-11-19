import streamlit as st
import pandas as pd
import sqlite3
import os
import hashlib
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import joblib
import xgboost
import requests
from datetime import datetime, timedelta
import base64
from PIL import Image
import io
import logging
import time
import threading
import streamlit.components.v1 as components

# === Config ===
st.set_page_config(page_title="🏠 House Finder Pro", layout="wide", initial_sidebar_state="expanded")
import yaml

# Load config from config.yaml
with open("config2.0.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

BASE_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_PATH, config["paths"]["dataset"])
DATABASE_NAME = os.path.join(BASE_PATH, "data", config["paths"]["database"])
MODEL_FILE = os.path.join(BASE_PATH, config["paths"]["model"])
UPLOAD_DIR = os.path.join(BASE_PATH, config["paths"]["uploads"])
DEFAULT_COORDINATES = [config["coordinates"]["default_lat"], config["coordinates"]["default_lon"]]

# Create necessary folders if they don't exist
os.makedirs(os.path.join(BASE_PATH, "data"), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ames, Iowa

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Logging setup
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === Database Setup ===
def init_db():
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, email TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS listings (
            id INTEGER PRIMARY KEY, user_id INTEGER, price REAL, bedrooms INTEGER, year_built INTEGER, 
            garage_cars INTEGER, lot_area INTEGER, overall_qual INTEGER, image_path TEXT, expires_at TEXT, 
            lat REAL, lon REAL, interest_count INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id))''')
        conn.commit()
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        st.error("Database initialization failed.")
    finally:
        conn.close()

init_db()

# === Language and Theme ===
language = st.sidebar.selectbox("🌐 Language", ["English", "O‘zbek", "Русский", "Español"])
theme = st.sidebar.selectbox("🎨 Theme", ["Light", "Dark"])
translations = {
    "English": {
        "welcome": "👋 Welcome to House Finder Pro!",
        "login": "Login",
        "register": "Register",
        "logout": "Logout",
        "search": "Search Houses",
        "sell": "Sell a House",
        "profile": "Profile",
        "announcements": "Announcements",
                "manual_login": "Manual Login",
        "search_button": "🔎 Search",
        "sell_button": "List House",
        "interest": "Show Interest",
        "map_view": "View on Map",
    },
    "O‘zbek": {
        "welcome": "👋 Uy Qidiruv Pro-ga xush kelibsiz!",
        "login": "Kirish",
        "register": "Ro‘yxatdan o‘tish",
        "logout": "Chiqish",
        "search": "Uylarni qidirish",
        "sell": "Uy sotish",
        "profile": "Profil",
        "announcements": "E’lonlar",
                "manual_login": "Qo‘lda kirish",
        "search_button": "🔎 Qidirish",
        "sell_button": "Uy qo‘shish",
        "interest": "Qiziqish bildirish",
        "map_view": "Xaritada ko‘rish",
    },
    "Русский": {
        "welcome": "👋 Добро пожаловать в House Finder Pro!",
        "login": "Вход",
        "register": "Регистрация",
        "logout": "Выход",
        "search": "Поиск домов",
        "sell": "Продать дом",
        "profile": "Профиль",
        "announcements": "Объявления",
                "manual_login": "Ручной вход",
        "search_button": "🔎 Искать",
        "sell_button": "Добавить дом",
        "interest": "Проявить интерес",
        "map_view": "Посмотреть на карте",
    },
    "Español": {
        "welcome": "👋 ¡Bienvenido a House Finder Pro!",
        "login": "Iniciar sesión",
        "register": "Registrarse",
        "logout": "Cerrar sesión",
        "search": "Buscar casas",
        "sell": "Vender una casa",
        "profile": "Perfil",
        "announcements": "Anuncios",
                "manual_login": "Inicio manual",
        "search_button": "🔎 Buscar",
        "sell_button": "Listar casa",
        "interest": "Mostrar interés",
        "map_view": "Ver en el mapa",
    }
}
t = translations[language]

# Apply theme
if theme == "Dark":
    st.markdown("""
        <style>
            body {background-color: #1e1e1e; color: #ffffff;}
            .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
            .card {background-color: #2e2e2e; padding: 10px; border-radius: 10px; margin: 10px 0;}
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body {background-color: #ffffff; color: #000000;}
            .stButton>button {background-color: #2196F3; color: white; border-radius: 5px;}
            .card {background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin: 10px 0;}
        </style>
    """, unsafe_allow_html=True)

# === Authentication ===
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", (username, hash_password(password), email))
        conn.commit()
        st.success(t["register"] + " successful!")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")
    except Exception as e:
        logging.error(f"Registration failed: {e}")
        st.error("Registration failed.")
    finally:
        conn.close()

def manual_login(username, password):
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, hash_password(password)))
        user = c.fetchone()
        conn.close()
        return user[0] if user else None
    except Exception as e:
        logging.error(f"Manual login failed: {e}")
        return None



if "user_id" not in st.session_state:
    st.session_state.user_id = None

# === Navigation ===
page = st.sidebar.radio("📑 Navigate", [t["login"], t["register"], t["search"], t["sell"], t["profile"], t["announcements"]], 
                        disabled=not st.session_state.user_id if "page" not in locals() else (not st.session_state.user_id and page not in [t["login"], t["register"]]))

# === Load Data and Model ===
@st.cache_data
def load_dataframe():
    try:
        return pd.read_csv(DATASET_PATH)
    except Exception as e:
        logging.error(f"Data loading failed: {e}")
        st.error("❌ Dataset not found.")
        return pd.DataFrame()

df = load_dataframe()

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except:
        if not df.empty:
            X = df[['Gr Liv Area', 'Bedroom AbvGr', 'Year Built', 'Garage Cars', 'Lot Area', 'Overall Qual']]
            y = df['SalePrice']
            model = xgboost.XGBRegressor(n_estimators=200)
            model.fit(X, y)
            joblib.dump(model, MODEL_FILE)
            return model
        return None

model = load_model()

# === Real-Time Updates ===
def poll_listings():
    while True:
        if st.session_state.user_id:
            try:
                conn = sqlite3.connect(DATABASE_NAME)
                expiring = pd.read_sql_query("SELECT * FROM listings WHERE user_id = ? AND expires_at <= ?", 
                                             (st.session_state.user_id, (datetime.now() + timedelta(days=3)).isoformat()))
                conn.close()
                if not expiring.empty:
                    st.session_state.notifications = f"🔔 {len(expiring)} listings expiring soon!"
                else:
                    st.session_state.notifications = None
            except Exception as e:
                logging.error(f"Polling failed: {e}")
        time.sleep(60)  # Poll every minute

if "notifications" not in st.session_state:
    st.session_state.notifications = None
    threading.Thread(target=poll_listings, daemon=True).start()

# === Pages ===
if page == t["login"]:
    st.title(t["login"])
    login_option = st.radio("Login Method", [t["manual_login"]])else:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button(t["login"]):
                user_id = manual_login(username, password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.success("Logged in successfully!")
                else:
                    st.error("Invalid credentials.")

elif page == t["register"]:
    st.title(t["register"])
    with st.form("register_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        email = st.text_input("Email")
        if st.form_submit_button(t["register"]):
            register_user(username, password, email)

elif page == t["search"]:
    st.title(t["search"])
    with st.form("search_form"):
        budget = st.number_input("Max Budget", value=250000)
        bedrooms = st.number_input("Bedrooms", value=3)
        year = st.number_input("Min Year Built", value=2000)
        garage = st.number_input("Garage Spaces", value=1)
        lot_size = st.number_input("Min Lot Area", value=5000)
        quality = st.slider("Min Overall Quality", 1, 10, 5)
        if st.form_submit_button(t["search_button"]):
            try:
                conn = sqlite3.connect(DATABASE_NAME)
                results = pd.read_sql_query("""
                    SELECT * FROM listings WHERE price <= ? AND bedrooms = ? AND year_built >= ? 
                    AND garage_cars = ? AND lot_area >= ? AND overall_qual >= ?
                """, conn, params=(budget, bedrooms, year, garage, lot_size, quality))
                conn.close()
                if not results.empty:
                    st.success(f"✅ Found {len(results)} matching houses")
                    # Map Visualization
                    m = folium.Map(location=DEFAULT_COORDINATES, zoom_start=12)
                    marker_cluster = MarkerCluster().add_to(m)
                    for _, row in results.iterrows():
                        folium.Marker([row["lat"] or DEFAULT_COORDINATES[0], row["lon"] or DEFAULT_COORDINATES[1]], 
                                      popup=f"${row['price']}").add_to(marker_cluster)
                    st_folium(m, width=700, height=500)
                    # Listings with Interest Button
                    for _, row in results.iterrows():
                        st.markdown(f"<div class='card'><b>Price:</b> ${row['price']} | <b>Interest:</b> {row['interest_count']}</div>", unsafe_allow_html=True)
                        if st.button(t["interest"], key=f"interest_{row['id']}"):
                            conn = sqlite3.connect(DATABASE_NAME)
                            c = conn.cursor()
                            c.execute("UPDATE listings SET interest_count = interest_count + 1 WHERE id = ?", (row["id"],))
                            conn.commit()
                            conn.close()
                            st.success("Interest recorded!")
                else:
                    st.warning("😕 No matches found.")
            except Exception as e:
                logging.error(f"Search failed: {e}")
                st.error("Search failed.")

elif page == t["sell"]:
    st.title(t["sell"])
    with st.form("sell_form"):
        price = st.number_input("Price", min_value=0)
        bedrooms = st.number_input("Bedrooms", min_value=0)
        year_built = st.number_input("Year Built", min_value=1900, max_value=2025)
        garage_cars = st.number_input("Garage Spaces", min_value=0)
        lot_area = st.number_input("Lot Area", min_value=0)
        overall_qual = st.slider("Overall Quality", 1, 10, 5)
        lat = st.number_input("Latitude", value=DEFAULT_COORDINATES[0])
        lon = st.number_input("Longitude", value=DEFAULT_COORDINATES[1])
        image = st.file_uploader("Upload House Image", type=["jpg", "png"])
        expires_in = st.slider("Listing Duration (days)", 1, 30, 7)
        if st.form_submit_button(t["sell_button"]):
            try:
                if image:
                    image_path = os.path.join(UPLOAD_DIR, f"{st.session_state.user_id}_{datetime.now().timestamp()}.jpg")
                    with open(image_path, "wb") as f:
                        f.write(image.read())
                conn = sqlite3.connect(DATABASE_NAME)
                c = conn.cursor()
                expires_at = (datetime.now() + timedelta(days=expires_in)).isoformat()
                c.execute("INSERT INTO listings (user_id, price, bedrooms, year_built, garage_cars, lot_area, overall_qual, image_path, expires_at, lat, lon) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (st.session_state.user_id, price, bedrooms, year_built, garage_cars, lot_area, overall_qual, image_path, expires_at, lat, lon))
                conn.commit()
                conn.close()
                st.success("House listed successfully!")
            except Exception as e:
                logging.error(f"Listing failed: {e}")
                st.error("Failed to list house.")

elif page == t["profile"]:
    st.title(t["profile"])
    if st.button(t["logout"]):
        st.session_state.user_id = None
        st.success("Logged out successfully!")

elif page == t["announcements"]:
    st.title(t["announcements"])
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        listings = pd.read_sql_query("SELECT * FROM listings WHERE expires_at > ?", (datetime.now().isoformat(),))
        conn.close()
        for _, listing in listings.iterrows():
            time_left = (datetime.fromisoformat(listing["expires_at"]) - datetime.now()).days
            st.markdown(f"<div class='card'><b>Price:</b> ${listing['price']} | <b>Expires in:</b> {time_left} days</div>", unsafe_allow_html=True)
    except Exception as e:
        logging.error(f"Announcements failed: {e}")
        st.error("Failed to load announcements.")

# === Notifications ===
if st.session_state.notifications:
    st.sidebar.markdown(f"<div style='background-color:#ffcc00;padding:10px;border-radius:5px;'>{st.session_state.notifications}</div>", unsafe_allow_html=True)

# === UI/UX Enhancements ===
st.markdown("""
    <style>
        .stForm {border: 1px solid #ccc; padding: 20px; border-radius: 10px;}
        .stSidebar {background-color: #f8f9fa;}
    </style>
""", unsafe_allow_html=True)