import os
import sqlite3
import uuid
import streamlit as st
from dotenv import load_dotenv

# ----------------------------------------------------
# 🔑 Load Environment Variables from .env File
# ----------------------------------------------------
load_dotenv()

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from tools import all_tools

# ----------------------------------------------------
# 🗄️ 1. SQLite Database Initialization
# ----------------------------------------------------
DB_FILE = "travel_memory.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
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
    conn.commit()
    conn.close()

init_db()

def get_profile_fact(category: str) -> str:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM user_profile WHERE category = ?", (category,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else ""

def save_profile_fact(category: str, new_value: str):
    existing = get_profile_fact(category)
    items = set([x.strip() for x in existing.split(",") if x.strip()])
    items.add(new_value)
    updated_str = ", ".join(items)
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO user_profile (category, value) VALUES (?, ?)", (category, updated_str))
    conn.commit()
    conn.close()

def save_chat_message(thread_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (thread_id, role, content) VALUES (?, ?, ?)", (thread_id, role, content))
    conn.commit()
    conn.close()

def get_chat_history(thread_id: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_history WHERE thread_id = ?", (thread_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]

def get_all_threads():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT thread_id FROM chat_history")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]

# ----------------------------------------------------
# 🤖 2. LangGraph Agent Setup
# ----------------------------------------------------
# ChatGroq will automatically pick up GROQ_API_KEY from environment variables loaded by dotenv
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

    return f"""
You are an intelligent, real-time AI Travel Assistant.

PERMANENT USER MEMORY:
- Visited Cities (NEVER recommend these again): [{visited}]
- Known Health Conditions: [{health}]
- Dietary Limits: [{diet}]

TOOL INVOCATION MANDATE:
1. When the user asks about WEATHER, you MUST invoke the `get_weather` tool. Do NOT guess weather or use prior memory.
2. When asked about ATTRACTIONS or FOOD, you MUST invoke the `get_places_and_food` tool.
3. When asked about FLIGHTS, TRAINS, or BUSES, you MUST invoke the `get_transit` tool.
4. When asked about LUGGAGE, CLOTHING, or PACKING, you MUST invoke the `get_packing_rules` tool.

Always deliver up-to-date factual data provided by your tools.
"""

# ----------------------------------------------------
# 🎨 3. Streamlit UI
# ----------------------------------------------------
st.set_page_config(page_title="AI Travel Assistant", page_icon="✈️", layout="wide")
st.title("✈️ AI Travel Assistant")

all_threads = get_all_threads()
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = all_threads[0] if all_threads else str(uuid.uuid4())[:8]

active_thread_id = st.session_state.active_thread_id

with st.sidebar:
    st.header("🗄️ Database Profile")
    st.write(f"**Visited Cities:** {get_profile_fact('visited') or 'None'}")
    st.write(f"**Health Notes:** {get_profile_fact('health') or 'None'}")
    st.write(f"**Diet Limits:** {get_profile_fact('diet') or 'None'}")

    st.write("---")
    st.header("💬 Saved Chat Threads")

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

user_input = st.chat_input("Ask about a trip or share travel facts...")

if user_input:
    st.chat_message("user").write(user_input)
    save_chat_message(active_thread_id, "user", user_input)

    text_low = user_input.lower()
    if "been to" in text_low or "visited" in text_low:
        words = user_input.split()
        for word in words:
            if word.istitle() and word not in ["I", "In", "The", "To", "A"]:
                save_profile_fact("visited", word)

    if "asthma" in text_low:
        save_profile_fact("health", "Asthma")
    if "diabetes" in text_low:
        save_profile_fact("health", "Diabetes")
    if "vegetarian" in text_low:
        save_profile_fact("diet", "Vegetarian")

    conn = sqlite3.connect("langgraph_checkpoints.db", check_same_thread=False)
    sqlite_checkpointer = SqliteSaver(conn)

    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=build_system_prompt(),
        checkpointer=sqlite_checkpointer
    )

    config = {"configurable": {"thread_id": active_thread_id}}

    with st.spinner("Calling live tools & updating response..."):
        response = agent.invoke(
            {"messages": [("user", user_input)]},
            config=config
        )
        
        print("\n--- AGENT STEP LOGS ---")
        for m in response["messages"]:
            if hasattr(m, "tool_calls") and m.tool_calls:
                print("🔨 TOOL CALLED:", m.tool_calls)

        ai_reply = response["messages"][-1].content

    st.chat_message("assistant").write(ai_reply)
    save_chat_message(active_thread_id, "assistant", ai_reply)
    conn.close()
    st.rerun()
