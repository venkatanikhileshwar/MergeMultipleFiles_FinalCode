# app.py
import streamlit as st
import importlib.util
import sys

# Function to load module from file
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load tab modules
file_to_db = load_module("file_to_db", "tabs/file_to_db.py")
multi_file_merge = load_module("multi_file_merge", "tabs/multi_file_merge.py")
db_to_db = load_module("db_to_db", "tabs/db_to_db.py")

# Page configuration
st.set_page_config(page_title="Enterprise Data Validation", layout="wide")
st.title("Enterprise Data Validation")

# Initialize session state
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = 0

# Create tabs
tab_list = ["File → DB", "Multi-File Merge → DB", "DB ↔ DB"]
tab1, tab2, tab3 = st.tabs(tab_list)

# Render each tab
with tab1:
    st.session_state["active_tab"] = 0
    file_to_db.render()

with tab2:
    st.session_state["active_tab"] = 1
    multi_file_merge.render()

with tab3:
    st.session_state["active_tab"] = 2
    db_to_db.render()