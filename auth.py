"""
auth.py
-------
User authentication using a simple JSON file.
No database required. Users stored in users.json
Passwords are hashed using SHA-256 for security.

Functions:
  - load_users()       : Read users.json
  - save_users()       : Write users.json
  - hash_password()    : SHA-256 hash
  - register_user()    : Add new user
  - login_user()       : Verify credentials
  - get_user()         : Get user by username
"""

import json, os, hashlib, time

def find_project():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    for name in ["FMS_PROJECT","flood_project","FLOOD_PROJECT","fms_project"]:
        p = os.path.join(desktop, name)
        if os.path.exists(p):
            return p
    return os.path.dirname(os.path.abspath(__file__))

BASE       = find_project()
USERS_FILE = os.path.join(BASE, "users.json")

# ─────────────────────────────────────────────────────────────
# DEFAULT USERS — created on first run
# ─────────────────────────────────────────────────────────────
DEFAULT_USERS = [
    {
        "username": "admin",
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "name":     "Administrator",
        "created":  time.strftime("%Y-%m-%d")
    },
    {
        "username": "officer",
        "password": hashlib.sha256("officer123".encode()).hexdigest(),
        "name":     "Relief Officer",
        "created":  time.strftime("%Y-%m-%d")
    }
]

# ─────────────────────────────────────────────────────────────
# LOAD / SAVE
# ─────────────────────────────────────────────────────────────
def load_users():
    """Load users from JSON file. Create default users if file missing."""
    if not os.path.exists(USERS_FILE):
        save_users(DEFAULT_USERS)
        print(f"  ✅ Created users.json with default accounts")
        return DEFAULT_USERS
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    """Save users list to JSON file."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.strip().encode()).hexdigest()

def get_user(username):
    """Return user dict if exists, else None."""
    users = load_users()
    for u in users:
        if u["username"].lower() == username.lower():
            return u
    return None

# ─────────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────────
def login_user(username, password):
    """
    Verify username and password.
    Returns (True, user_dict) on success.
    Returns (False, error_message) on failure.
    """
    if not username or not password:
        return False, "Please enter both username and password."

    user = get_user(username)
    if not user:
        return False, "Username not found. Please check and try again."

    if user["password"] != hash_password(password):
        return False, "Incorrect password. Please try again."

    return True, user

# ─────────────────────────────────────────────────────────────
# REGISTER
# ─────────────────────────────────────────────────────────────
def register_user(username, password, confirm_password, name):
    """
    Register a new user.
    Returns (True, success_message) or (False, error_message).
    """
    # Validation
    if not username or not password or not name:
        return False, "All fields are required."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if password != confirm_password:
        return False, "Passwords do not match."
    if " " in username:
        return False, "Username cannot contain spaces."

    # Check if username already exists
    if get_user(username):
        return False, f"Username '{username}' already exists. Please choose another."

    # Add new user
    users = load_users()
    users.append({
        "username": username.lower(),
        "password": hash_password(password),
        "name":     name.strip(),
        "created":  time.strftime("%Y-%m-%d")
    })
    save_users(users)
    return True, f"Account created successfully! You can now log in as '{username}'."

# ─────────────────────────────────────────────────────────────
# LIST ALL USERS (for admin)
# ─────────────────────────────────────────────────────────────
def list_users():
    """Return all users without passwords."""
    users = load_users()
    return [{"username": u["username"], "name": u["name"],
             "created": u["created"]} for u in users]

# ─────────────────────────────────────────────────────────────
# DELETE USER
# ─────────────────────────────────────────────────────────────
def delete_user(username):
    """Delete a user by username. Cannot delete 'admin'."""
    if username.lower() == "admin":
        return False, "Cannot delete the admin account."
    users = load_users()
    new_users = [u for u in users if u["username"].lower() != username.lower()]
    if len(new_users) == len(users):
        return False, f"User '{username}' not found."
    save_users(new_users)
    return True, f"User '{username}' deleted successfully."