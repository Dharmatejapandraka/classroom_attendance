# 🎓 FaceTrack Classroom — Multi-Class Attendance App

A Streamlit app for schools, extended from FaceTrack, that lets you:
- Log in as **Admin** (sees/manages every class) or **Teacher** (scoped to their own class only)
- Register a student with name + phone + **Class/Section**, captured via face photos
- Take attendance by scanning a face with the camera (auto-marks present, no duplicates same day;
  a teacher session rejects a match that belongs to a different class)
- View a per-class or all-classes dashboard: totals, present/absent, a pie chart, and a 7-day trend
- **Export a PDF attendance report for any single class** for the current day
- Manage (delete) students, scoped by class
- Admins create/remove **Teacher accounts**, each tied to one class/section

Built entirely with Streamlit + OpenCV so it deploys with zero native/dlib compilation.

## What changed from the base FaceTrack app

| Area | Change |
|---|---|
| `database.py` | Added `class_section` column on `members`; added `teachers` table; every read/write helper now takes an optional `class_section` filter |
| `app.py` | Login page now has Admin / Teacher tabs; every page (Dashboard, Register, Attendance, Manage) is class-scoped for teachers, class-filterable for admins; new **Manage Teachers** page (admin-only) |
| `pdf_report.py` | **New file** — builds a per-class, per-day PDF report (summary + present/absent tables) with `reportlab` |
| `face_utils.py` | **Unchanged** — face detection/recognition is class-agnostic; a student's face is trained against their global `member_id` regardless of class |

## How it works

- **Detection**: OpenCV Haar Cascade finds the face in the camera frame.
- **Recognition**: OpenCV's LBPH (Local Binary Patterns Histograms) recognizer is trained on
  the saved face crops for each student and re-trained every time someone is added or removed.
- **Storage**: SQLite (`data/attendance.db`) for students, classes, attendance, and teacher accounts.
  Face image crops live under `data/faces/<member_id>/`.
- **Camera**: Uses `st.camera_input`, which pulls from the *browser's* camera — this is what
  makes it work both locally and on Streamlit Community Cloud.
- **PDF reports**: Generated on the fly with `reportlab` when an admin or teacher clicks
  **Export Attendance Report (PDF)** on the Dashboard, scoped to the class currently selected.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Default admin login: **Dharmateja / Dharmateja@1234** (change via secrets, see below).
Teacher logins don't exist until an admin creates them under **Manage Teachers**.

To change the admin login, create `.streamlit/secrets.toml` (copy `secrets.toml.example`) with:

```toml
[credentials]
username = "your_admin"
password = "your_strong_password"
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (make sure `data/` is empty/ignored — it gets created at runtime).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**, point it at your repo, main file `app.py`.
3. In the app's **Settings → Secrets**, paste your admin credentials as above.
4. Deploy. `packages.txt` will automatically install the system libraries OpenCV needs.

> Note: Streamlit Community Cloud's filesystem is ephemeral — if the app restarts/reboots,
> registered students, teacher accounts, and attendance history will reset. For permanent
> production use, swap the SQLite file and `data/faces` folder for a persistent store
> (e.g. S3 for images + a hosted Postgres database).

## Typical workflow

1. Admin logs in, goes to **Manage Teachers**, creates a teacher account tied to e.g. "Grade 8 - A".
2. Admin or the teacher goes to **Register Face** and enrolls students into that class.
3. Each day, the teacher logs in, opens **Take Attendance**, and scans faces as students arrive.
4. At the end of the day (or any time), the teacher opens **Dashboard** and clicks
   **Export Grade 8 - A Attendance Report (PDF)** to download a report for that class/day.
5. Admin can switch the Dashboard's class filter to "All Classes" for a school-wide view.

## Tuning recognition accuracy

In `face_utils.py`:
- `CONFIDENCE_THRESHOLD` (default 75) — lower = stricter matching (fewer false positives, more false "not recognized"). Raise it a bit if real students keep getting rejected; lower it if strangers get matched.
- Capture 4-5 photos per student from slightly different angles/lighting during registration for best accuracy.

## File structure

```
classroom_attendance_app/
├── app.py                 # Streamlit UI (login, dashboard, registration, attendance, management)
├── database.py             # SQLite helpers (members + classes + attendance + teacher accounts)
├── face_utils.py            # Face detection + LBPH training/recognition (unchanged from base app)
├── pdf_report.py            # Per-class PDF attendance report generator (new)
├── requirements.txt
├── packages.txt             # system deps for OpenCV on Streamlit Cloud
├── .streamlit/secrets.toml.example
└── data/                   # created at runtime (db + face images)
```
