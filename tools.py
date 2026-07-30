import os
import requests
import sqlite3
from typing import List
from pydantic import BaseModel, Field
from tavily import TavilyClient

from langchain_groq import ChatGroq
from langchain_core.tools import Tool

DB_FILE = "travel_memory.db"

class UserMemoryExtraction(BaseModel):
    visited_places: List[str] = Field(default=[], description="Cities/countries the user has explicitly visited")
    health_conditions: List[str] = Field(default=[], description="Medical conditions, allergies, or health restrictions")
    dietary_preferences: List[str] = Field(default=[], description="Dietary restrictions or preferences e.g. Vegan, Vegetarian")

def get_profile_fact(category: str) -> str:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM user_profile WHERE category = ?", (category,))
        row = cursor.fetchone()
        return row[0] if row else ""

def save_profile_fact(category: str, new_value: str):
    existing = get_profile_fact(category)
    items = set([x.strip() for x in existing.split(",") if x.strip()])
    items.add(new_value)
    updated_str = ", ".join(items)
    
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO user_profile (category, value) VALUES (?, ?)", (category, updated_str))

def extract_and_store_user_memory(user_text: str) -> str:
    extractor_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)
    structured_extractor = extractor_llm.with_structured_output(UserMemoryExtraction)
    
    try:
        data: UserMemoryExtraction = structured_extractor.invoke(
            f"Extract personal facts from this message if present: '{user_text}'"
        )
        for place in data.visited_places:
            save_profile_fact("visited", place.title())
        for health in data.health_conditions:
            save_profile_fact("health", health.title())
        for diet in data.dietary_preferences:
            save_profile_fact("diet", diet.title())
            
        return "User memory profile successfully synchronized with database."
    except Exception as e:
        return f"Memory extraction error: {str(e)}"

def get_weather(city: str) -> str:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OPENWEATHER_API_KEY environment variable is missing."
        
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city.strip()}&appid={api_key}&units=metric"
    
    try:
        res = requests.get(url, timeout=8).json()
        if res.get("cod") == 200:
            temp = res["main"]["temp"]
            desc = res["weather"][0]["description"]
            humidity = res["main"]["humidity"]
            return f"Weather in {city.strip().title()}: {temp}°C, {desc}, Humidity: {humidity}%."
        return f"Could not retrieve weather for '{city}'. Reason: {res.get('message', 'City not found')}."
    except Exception as e:
        return f"Weather service network exception: {str(e)}"

def get_places_and_food(destination: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is missing."
    try:
        client = TavilyClient(api_key=api_key)
        res = client.search(query=f"top must-visit tourist attractions and famous local food in {destination}", max_results=4)
        results = res.get("results", [])
        if results:
            return "\n\n".join([f"- {item['title']}: {item['content']}" for item in results])
        return f"No search results returned for {destination}."
    except Exception as e:
        return f"Search service exception: {str(e)}"

def get_transit(route: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is missing."
    try:
        client = TavilyClient(api_key=api_key)
        res = client.search(query=f"travel options flights trains buses routes for {route}", max_results=4)
        results = res.get("results", [])
        if results:
            return "\n\n".join([f"- {item['title']}: {item['content']}" for item in results])
        return f"No transit details found for {route}."
    except Exception as e:
        return f"Transit service exception: {str(e)}"

def generate_packing_tips(destination_and_weather_info: str) -> str:
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)
    health_notes = get_profile_fact("health") or "None"
    
    prompt = f"""
    Destination & Weather Context: '{destination_and_weather_info}'
    User Health Profile: '{health_notes}'
    
    Task: Generate EXACTLY 5 practical, dynamic luggage packing tips tailored specifically to the climate and health facts above.
    Do NOT output generic templates.
    """
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Packing tip generation error: {str(e)}"

all_tools = [
    Tool(
        name="get_weather", 
        func=get_weather, 
        description="Retrieves real-time weather from OpenWeather. Parameter must be a valid specific city name (e.g. 'Srinagar', 'Kochi', 'Hyderabad', 'Paris')."
    ),
    Tool(
        name="get_places_and_food", 
        func=get_places_and_food, 
        description="Searches real-time web data for top tourist attractions and local food specialties for a destination."
    ),
    Tool(
        name="get_transit", 
        func=get_transit, 
        description="Searches travel options (flights, trains, buses) between origin and destination routes."
    ),
    Tool(
        name="get_packing_recommendations", 
        func=generate_packing_tips, 
        description="Generates 5 dynamic packing recommendations based on weather and user health notes."
    ),
    Tool(
        name="extract_user_memory", 
        func=extract_and_store_user_memory, 
        description="Extracts user personal facts (visited places, health conditions, dietary habits) and persists them in SQLite."
    )
]
