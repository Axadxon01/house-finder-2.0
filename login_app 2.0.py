# === Authentication ===
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_user_db():
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                password TEXT
            )
        """)
        conn.commit()
    except Exception as e:
        logging.error(f"User DB initialization failed: {e}")
    finally:
        conn.close()

def register_user(username, password):
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        st.success("Registered successfully!")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")
    except Exception as e:
        logging.error(f"Registration error: {e}")
        st.error("Registration failed.")
    finally:
        conn.close()

def manual_login(username, password):
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, hash_password(password)))
        user = c.fetchone()
        return user[0] if user else None
    except Exception as e:
        logging.error(f"Login error: {e}")
        return None
    finally:
        conn.close()

if "user_id" not in st.session_state:
    st.session_state.user_id = None

init_user_db()

st.sidebar.subheader("🔐 Login / Register")
auth_mode = st.sidebar.radio("Choose action", ["Login", "Register"])
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if auth_mode == "Login":
    if st.sidebar.button("Login"):
        user_id = manual_login(username, password)
        if user_id:
            st.session_state.user_id = user_id
            st.sidebar.success("Logged in!")
        else:
            st.sidebar.error("Invalid credentials.")
else:
    if st.sidebar.button("Register"):
        register_user(username, password)
