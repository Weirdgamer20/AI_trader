@echo off
setlocal
if not exist .venv (
  py -3.12 -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
