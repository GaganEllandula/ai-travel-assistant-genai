import os
import requests
from tavily import TavilyClient
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.tools import Tool


# 1. RAG Vector Store (Packing & Baggage Rules)

docs = [
    Document(
        page_content="""
        PACKING & HEALTH ADVICE:
        - Carry-on baggage: Liquids under 100ml. Batteries, laptops, power banks in carry-on bags.
        - Hot Weather: Breathable light cotton clothes, SPF 50 sunscreen, hydration electrolytes, sunglasses.
        - Cold Weather: Layered thermal innerwear, fleece jackets, wool socks.
        - Monsoon/Rain: Quick-dry apparel, rain jackets, waterproof footwear.
        - Health Precautions: Carry required prescription medicines, basic first-aid, and anti-allergy pills.
        """
    )
]

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)

def get_packing_rules(query: str) -> str:
    results = vectorstore.similarity_search(query, k=1)
    return results[0].page_content if results else "Pack light cotton clothing and general medication."

# 2. Weather Tool (OpenWeather API)
def get_weather(city: str) -> str:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    # Clean and alias city names (OpenWeather expects 'Kochi' instead of 'Kochin')
    clean_city = city.strip().replace("Kochin", "Kochi")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={clean_city}&appid={api_key}&units=metric"
    
    try:
        res = requests.get(url, timeout=5).json()
        if res.get("cod") == 200:
            temp = res["main"]["temp"]
            desc = res["weather"][0]["description"]
            humidity = res["main"]["humidity"]
            return f"LIVE REAL-TIME WEATHER IN {city.upper()}: {temp}°C, {desc}, Humidity: {humidity}%."
        else:
            print(f"[DEBUG] Weather API Error: {res}")
            return f"Error fetching live weather for {city}: {res.get('message', 'City not found')}"
    except Exception as e:
        print(f"[DEBUG] Weather Exception: {str(e)}")
        return f"Could not fetch weather due to network issue: {str(e)}"

# 3. Web Search Tool for Places & Food (Tavily API)

def get_places_and_food(city: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    try:
        client = TavilyClient(api_key=api_key)
        res = client.search(query=f"top must-visit places and iconic local food in {city}", max_results=3)
        results = res.get("results", [])
        if results:
            return "\n\n".join([f"- {item['title']}: {item['content']}" for item in results])
        return f"No live Tavily results found for {city}."
    except Exception as e:
        print(f"[DEBUG] Tavily Places Exception: {str(e)}")
        return f"Search error for {city}: {str(e)}"

# 4. Web Search Tool for Transit & Routes (Tavily API)

def get_transit(route: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    try:
        client = TavilyClient(api_key=api_key)
        res = client.search(query=f"flights trains buses travel options for {route}", max_results=3)
        results = res.get("results", [])
        if results:
            return "\n\n".join([f"- {item['title']}: {item['content']}" for item in results])
        return f"No live transit search results found for {route}."
    except Exception as e:
        print(f"[DEBUG] Tavily Transit Exception: {str(e)}")
        return f"Transit search error: {str(e)}"

# Export LangChain tools
all_tools = [
    Tool(
        name="get_weather", 
        func=get_weather, 
        description="MUST BE USED to fetch live real-time weather, temperature, and forecast for any destination city."
    ),
    Tool(
        name="get_places_and_food", 
        func=get_places_and_food, 
        description="MUST BE USED to find top tourist places, attractions, and local food dishes for any city."
    ),
    Tool(
        name="get_transit", 
        func=get_transit, 
        description="MUST BE USED to find flight, train, or bus options between origin and destination cities."
    ),
    Tool(
        name="get_packing_rules", 
        func=get_packing_rules, 
        description="MUST BE USED to retrieve baggage rules, health tips, and clothing guidelines."
    )
]
