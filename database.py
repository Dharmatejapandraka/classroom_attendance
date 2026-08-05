"""
database.py
SQLite persistence layer for classes, members, attendance records, and teacher accounts.
"""

import sqlite3
import os
from datetime import datetime, date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "attendance.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            class_section TEXT NOT NULL DEFAULT 'Unassigned',
            created_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            FOREIGN KEY (member_id) REFERENCES members (id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            class_section TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------------------
# Members
# --------------------------------------------------------------------------------------
def add_member(name, phone, class_section):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO members (name, phone, class_section, created_at) VALUES (?, ?, ?, ?)",
        (name, phone, class_section, datetime.now().isoformat()),
    )
    member_id = c.lastrowid
    conn.commit()
    conn.close()
    return member_id


def phone_exists(phone):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM members WHERE phone=?", (phone,))
    row = c.fetchone()
    conn.close()
    return row is not None


def get_member_name(member_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT name FROM members WHERE id=?", (member_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def get_all_members(class_section=None):
    """Returns list of tuples: (id, name, phone, class_section).
    If class_section is given, only members in that class are returned."""
    conn = get_connection()
    c = conn.cursor()
    if class_section:
        c.execute(
            "SELECT id, name, phone, class_section FROM members WHERE class_section=? ORDER BY name",
            (class_section,),
        )
    else:
        c.execute("SELECT id, name, phone, class_section FROM members ORDER BY class_section, name")
    rows = c.fetchall()
    conn.close()
    return rows


def get_all_classes():
    """Returns sorted list of distinct class/section names currently in use."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT DISTINCT class_section FROM members ORDER BY class_section")
    rows = [r[0] for r in c.fetchall()]
    conn.close()
    return rows


def delete_member(member_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM members WHERE id=?", (member_id,))
    c.execute("DELETE FROM attendance WHERE member_id=?", (member_id,))
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------------------
# Attendance
# --------------------------------------------------------------------------------------
def mark_attendance(member_id):
    """Marks attendance for today if not already marked. Returns True if newly marked."""
    today = date.today().isoformat()
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM attendance WHERE member_id=? AND date=?", (member_id, today))
    existing = c.fetchone()
    if existing:
        conn.close()
        return False
    now_time = datetime.now().strftime("%H:%M:%S")
    c.execute(
        "INSERT INTO attendance (member_id, date, time) VALUES (?, ?, ?)",
        (member_id, today, now_time),
    )
    conn.commit()
    conn.close()
    return True


def get_today_present_ids(class_section=None):
    """Returns dict {member_id: time_str} for members marked present today,
    optionally scoped to a single class_section."""
    today = date.today().isoformat()
    conn = get_connection()
    c = conn.cursor()
    if class_section:
        c.execute(
            """SELECT a.member_id, a.time FROM attendance a
               JOIN members m ON m.id = a.member_id
               WHERE a.date=? AND m.class_section=?""",
            (today, class_section),
        )
    else:
        c.execute("SELECT member_id, time FROM attendance WHERE date=?", (today,))
    rows = c.fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def get_attendance_history(days=7, class_section=None):
    """Returns list of (date, present_count) for the last N days that have records,
    optionally scoped to a single class_section."""
    conn = get_connection()
    c = conn.cursor()
    if class_section:
        c.execute(
            """SELECT a.date, COUNT(DISTINCT a.member_id) FROM attendance a
               JOIN members m ON m.id = a.member_id
               WHERE m.class_section=?
               GROUP BY a.date ORDER BY a.date DESC LIMIT ?""",
            (class_section, days),
        )
    else:
        c.execute(
            """SELECT date, COUNT(DISTINCT member_id) FROM attendance
               GROUP BY date ORDER BY date DESC LIMIT ?""",
            (days,),
        )
    rows = c.fetchall()
    conn.close()
    return list(reversed(rows))


def get_attendance_on_date(target_date, class_section=None):
    """Returns dict {member_id: time_str} for members marked present on a specific date
    (ISO 'YYYY-MM-DD'), optionally scoped to a single class_section. Used for PDF export."""
    conn = get_connection()
    c = conn.cursor()
    if class_section:
        c.execute(
            """SELECT a.member_id, a.time FROM attendance a
               JOIN members m ON m.id = a.member_id
               WHERE a.date=? AND m.class_section=?""",
            (target_date, class_section),
        )
    else:
        c.execute("SELECT member_id, time FROM attendance WHERE date=?", (target_date,))
    rows = c.fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


# --------------------------------------------------------------------------------------
# Teacher accounts (admin manages these; teachers log in with them)
# --------------------------------------------------------------------------------------
def add_teacher(username, password, name, class_section):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO teachers (username, password, name, class_section, created_at) VALUES (?, ?, ?, ?, ?)",
        (username, password, name, class_section, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def username_exists(username):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM teachers WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    return row is not None


def get_teacher_by_login(username, password):
    """Returns (id, name, class_section) if credentials match, else None."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id, name, class_section FROM teachers WHERE username=? AND password=?",
        (username, password),
    )
    row = c.fetchone()
    conn.close()
    return row


def get_all_teachers():
    """Returns list of tuples: (id, username, name, class_section)"""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, username, name, class_section FROM teachers ORDER BY class_section, name")
    rows = c.fetchall()
    conn.close()
    return rows


def delete_teacher(teacher_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM teachers WHERE id=?", (teacher_id,))
    conn.commit()
    conn.close()
