import streamlit as st
import pandas as pd
import sqlite3
import os
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import joblib
import xgboost
from datetime import datetime, timedelta
import logging
from PIL import Image

# === Config ===
st.set_page_config(page_title="🏠 House Finder Pro", layout="wide", initial_sidebar_state="expanded")
BASE_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_PATH, "AmesHousing.csv")
DATABASE_NAME = os.path.join(BASE_PATH, "houses.db")  # Use BASE_PATH instead of /data
MODEL_FILE = os.path.join(BASE_PATH, "house_price_model.pkl")
UPLOAD_DIR = os.path.join(BASE_PATH, "uploads")
DEFAULT_COORDINATES = [42.0347, -93.6200]  # Ames, Iowa

# Ensure directories
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Logging setup
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === Initialize Database ===
def init_db():
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS listings (
            id INTEGER PRIMARY KEY,
            price REAL,
            bedrooms INTEGER,
            year_built INTEGER,
            garage_cars INTEGER,
            lot_area INTEGER,
            overall_qual INTEGER,
            image_path TEXT,
            expires_at TEXT,
            lat REAL,
            lon REAL,
            interest_count INTEGER DEFAULT 0
        )''')
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        st.error("Database initialization failed.")

init_db()

# === Load Dataset ===
@st.cache_data
def load_dataframe():
    try:
        return pd.read_csv(DATASET_PATH)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
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

# === Language and Theme Settings ===
if "language" not in st.session_state:
    st.session_state.language = "English"
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

language = st.sidebar.selectbox("🌐 Language", ["English", "O‘zbek", "Русский", "Español"], index=["English", "O‘zbek", "Русский", "Español"].index(st.session_state.language))
theme = st.sidebar.selectbox("🎨 Theme", ["Light", "Dark"], index=["Light", "Dark"].index(st.session_state.theme))

st.session_state.language = language
st.session_state.theme = theme

translations = {
    "English": {
        "search": "Search Houses",
        "sell": "Sell a House",
        "profile": "Profile",
        "announcements": "Announcements",
        "search_button": "🔎 Search",
        "sell_button": "List House",
        "interest": "Show Interest"
    },
    "O‘zbek": {
        "search": "Uylarni qidirish",
        "sell": "Uy sotish",
        "profile": "Profil",
        "announcements": "E’lonlar",
        "search_button": "🔎 Qidirish",
        "sell_button": "Uy qo‘shish",
        "interest": "Qiziqish bildirish"
    },
    "Русский": {
        "search": "Поиск домов",
        "sell": "Продать дом",
        "profile": "Профиль",
        "announcements": "Объявления",
        "search_button": "🔎 Искать",
        "sell_button": "Добавить дом",
        "interest": "Проявить интерес"
    },
    "Español": {
        "search": "Buscar casas",
        "sell": "Vender una casa",
        "profile": "Perfil",
        "announcements": "Anuncios",
        "search_button": "🔎 Buscar",
        "sell_button": "Listar casa",
        "interest": "Mostrar interés"
    }
}
t = translations[language]

# === Navigation ===
if "page" not in st.session_state:
    st.session_state.page = t["search"]

page = st.sidebar.radio("📑 Navigate", [t["search"], t["sell"], t["profile"], t["announcements"]], index=0)

# === Pages ===
if page == t["search"]:
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
                    m = folium.Map(location=DEFAULT_COORDINATES, zoom_start=12)
                    marker_cluster = MarkerCluster().add_to(m)
                    for _, row in results.iterrows():
                        folium.Marker([row["lat"], row["lon"]], popup=f"${row['price']}").add_to(marker_cluster)
                    st_folium(m, width=700, height=500)
                    for _, row in results.iterrows():
                        st.markdown(f"<div style='padding:10px;border:1px solid #ddd;border-radius:10px;margin-bottom:10px'><b>Price:</b> ${row['price']} | <b>Interest:</b> {row['interest_count']}</div>", unsafe_allow_html=True)
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
                image_path = ""
                if image:
                    image_path = os.path.join(UPLOAD_DIR, f"{datetime.now().timestamp()}.jpg")
                    with open(image_path, "wb") as f:
                        f.write(image.read())
                expires_at = (datetime.now() + timedelta(days=expires_in)).isoformat()
                conn = sqlite3.connect(DATABASE_NAME)
                c = conn.cursor()
                c.execute("""INSERT INTO listings (price, bedrooms, year_built, garage_cars, lot_area, 
                             overall_qual, image_path, expires_at, lat, lon)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                          (price, bedrooms, year_built, garage_cars, lot_area, overall_qual, image_path, expires_at, lat, lon))
                conn.commit()
                conn.close()
                st.success("House listed successfully!")
            except Exception as e:
                logging.error(f"Listing failed: {e}")
                st.error("Failed to list house.")

elif page == t["profile"]:
    st.title(t["profile"])
    st.write("Welcome to your profile!")

elif page == t["announcements"]:
    st.title(t["announcements"])
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        listings = pd.read_sql_query("SELECT * FROM listings WHERE expires_at > ?", (datetime.now().isoformat(),))
        conn.close()
        for _, listing in listings.iterrows():
            time_left = (datetime.fromisoformat(listing["expires_at"]) - datetime.now()).days
            st.markdown(f"<div style='padding:10px;border:1px solid #ccc;border-radius:10px;margin-bottom:10px'><b>Price:</b> ${listing['price']} | <b>Expires in:</b> {time_left} days</div>", unsafe_allow_html=True)
    except Exception as e:
        logging.error(f"Announcements failed: {e}")
        st.error("Failed to load announcements.")
