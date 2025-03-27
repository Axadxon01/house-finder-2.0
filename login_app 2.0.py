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
