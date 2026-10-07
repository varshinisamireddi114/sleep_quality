import sqlite3
import bcrypt

DB_PATH = "sleepcare.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def register_user(name, email, password):
    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    try:
        conn = get_connection()

        conn.execute(
            """
            INSERT INTO users (name, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (name, email, password_hash)
        )

        conn.commit()
        conn.close()

        return True, "Registration successful."

    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."


def login_user(email, password):
    conn = get_connection()

    user = conn.execute(
        """
        SELECT id, name, email, password_hash
        FROM users
        WHERE email = ?
        """,
        (email,)
    ).fetchone()

    conn.close()

    if user is None:
        return False, None

    stored_hash = user[3].encode("utf-8")

    if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
        return True, {
            "id": user[0],
            "name": user[1],
            "email": user[2]
        }

    return False, None