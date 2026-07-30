# ai-travel-assistant-genai
An intelligent, multi-turn AI Travel Assistant built with Streamlit, LangGraph, and LangChain. The chatbot acts as an interactive travel agent that utilizes real-time API tools to fetch weather, transportation options, tourist places, and food recommendations while preserving persistent long-term user memory and chat history using SQLite.

==>Key Features

->Persistent User Memory Profile: Remembers user-specific details across sessions (visited cities, medical conditions like Asthma/Diabetes, and dietary preferences like Vegetarian/Vegan) so it never repeats past trips and tailors recommendations safely.

->Multi-Thread Chat Management: Complete conversational history saved per thread using SQLite. You can seamlessly switch between past chat sessions or start new ones.

==>Dynamic Real-Time Tools:

->OpenWeather API: Real-time weather, temperature, and humidity checks for any destination.

->Tavily Web Search API: Up-to-date queries for top tourist attractions, local food specialties, and transit routes (flights, trains, buses).

->Local RAG Vector Store: Uses CPU-friendly FastEmbed and InMemoryVectorStore to query baggage limits, packing rules, and travel health guidelines.

->Robust ReAct Agent: Powered by llama-3.3-70b-versatile via Groq for high-speed inference with automatic retries and timeout protection.
