import streamlit as st
import sqlite3
import hashlib
import os

# === Config ===
st.set_page_config(page_title="🔐 Login - House Finder", layout="centered")
BASE_PATH = os.path.dirname(__file__)
DATABASE_NAME = os.path.join(BASE_PATH, "houses.db")

# === Database Setup ===
def init_db():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, email TEXT)''')
    conn.commit()
    conn.close()

init_db()

# === Authentication Functions ===
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", (username, hash_password(password), email))
        conn.commit()
        st.success("Registration successful! Please log in.")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")
    finally:
        conn.close()

def login_user(username, password):
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, hash_password(password)))
        user = c.fetchone()
        conn.close()
        return user[0] if user else None
    except Exception as e:
        st.error("An error occurred during login.")
        return None

# === UI ===
st.title("🔐 Login to House Finder")

if "user_id" not in st.session_state:
    st.session_state.user_id = None

tab1, tab2 = st.tabs(["Login", "Register"])

with tab1:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            user_id = login_user(username, password)
            if user_id:
                st.session_state.user_id = user_id
                st.success("Logged in successfully! Redirecting to main app...")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials.")

with tab2:
    with st.form("register_form"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        email = st.text_input("Email")
        if st.form_submit_button("Register"):
            register_user(new_username, new_password, email)

if st.session_state.user_id:
    st.write("You are logged in! Navigate to the main app.")