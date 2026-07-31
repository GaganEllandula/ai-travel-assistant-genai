import os
import sqlite3
import uuid
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from tools import all_tools, get_profile_fact, extract_and_store_user_memory

DB_FILE = "travel_memory.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                category TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                role TEXT,
                content TEXT
            )
        """)

init_db()

def save_chat_message(thread_id: str, role: str, content: str):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO chat_history (thread_id, role, content) VALUES (?, ?, ?)", (thread_id, role, content))

def get_chat_history(thread_id: str):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role, content FROM chat_history WHERE thread_id = ?", (thread_id,))
        rows = cursor.fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

def get_all_threads():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM chat_history")
        rows = cursor.fetchall()
        return [r[0] for r in rows]

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile", 
    temperature=0.1,
    request_timeout=60.0,
    max_retries=3
)

def build_system_prompt():
    visited = get_profile_fact("visited") or "None"
    health = get_profile_fact("health") or "None"
    diet = get_profile_fact("diet") or "None"
    prefs = get_profile_fact("preferences") or "None"

    return f"""
You are an enterprise-grade, real-time AI Travel Agent.

USER DATABASE PROFILE:
- Previously Visited Locations (DO NOT recommend visiting these): [{visited}]
- Health Notes / Restrictions: [{health}]
- Dietary Preferences: [{diet}]
- General Preferences & Personal Notes: [{prefs}]

CRITICAL EXECUTION RULES:
1. ALWAYS execute tools to gather real-time data before answering travel queries.
2. For weather queries (`get_weather`), derive and pass the specific hub city name (e.g., pass 'Srinagar' for Kashmir, 'Kochi' for Kerala, 'Panaji' for Goa). Do NOT pass full state or region names into `get_weather`.
3. ALWAYS execute `get_transit` with origin and destination routes and explicitly detail modes of transport in Section 2.
4. Execute `get_places_and_food` for attractions and local cuisine.
5. Execute `get_packing_recommendations` to dynamically produce 5 packing tips based on weather and health/preference notes.

OUTPUT FORMAT STANDARD:
Your final response MUST be comprehensive, structured, and contain all 7 of the following sections:
- 🗺️ 1. Ideal Duration & Day-by-Day Itinerary
- 🚆 2. Transportation Options & Availability (Explicitly list Flights, Trains, and Buses with travel duration, estimated fares, and frequencies)
- 🍛 3. Attractions & Local Cuisine
- 🌦️ 4. Current Weather & Best Time to Visit (incorporates real-time weather tool output)
- 💰 5. Estimated Budget Breakdown
- 🧳 6. Dynamic Luggage & Packing Tips (exactly 5 items from tool output)
- 🛡️ 7. Safety & Travel Precautions
"""

st.set_page_config(page_title="AI Travel Assistant", page_icon="✈️", layout="wide")
st.title("✈️ AI Personal Travel Assistant")

all_threads = get_all_threads()
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = all_threads[0] if all_threads else str(uuid.uuid4())[:8]

active_thread_id = st.session_state.active_thread_id

with st.sidebar:
    st.header("🗄️ Database User Profile")
    st.write(f"**Visited Places:** {get_profile_fact('visited') or 'None'}")
    st.write(f"**Health Notes:** {get_profile_fact('health') or 'None'}")
    st.write(f"**Dietary Preferences:** {get_profile_fact('diet') or 'None'}")
    st.write(f"**General Preferences:** {get_profile_fact('preferences') or 'None'}")

    st.write("---")
    st.header("💬 Chat Threads")

    if st.button("+ New Chat Thread", use_container_width=True):
        new_thread = str(uuid.uuid4())[:8]
        st.session_state.active_thread_id = new_thread
        st.rerun()

    for tid in get_all_threads():
        btn_label = f"Thread {tid}"
        if tid == active_thread_id:
            btn_label = f"👉 {btn_label}"
        if st.button(btn_label, key=tid, use_container_width=True):
            st.session_state.active_thread_id = tid
            st.rerun()

messages_from_db = get_chat_history(active_thread_id)
for msg in messages_from_db:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Plan a trip or update personal facts...")

if user_input:
    st.chat_message("user").write(user_input)
    save_chat_message(active_thread_id, "user", user_input)

    with st.spinner("Executing real-time tools and compiling itinerary..."):
        extract_and_store_user_memory(user_input)

        conn = sqlite3.connect("langgraph_checkpoints.db", check_same_thread=False)
        sqlite_checkpointer = SqliteSaver(conn)

        agent = create_react_agent(
            model=llm,
            tools=all_tools,
            prompt=build_system_prompt(),
            checkpointer=sqlite_checkpointer
        )

        config = {"configurable": {"thread_id": active_thread_id}}

        response = agent.invoke(
            {"messages": [("user", user_input)]},
            config=config
        )
        
        ai_reply = response["messages"][-1].content
        conn.close()

    st.chat_message("assistant").write(ai_reply)
    save_chat_message(active_thread_id, "assistant", ai_reply)
    st.rerun()
