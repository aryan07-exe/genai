import sqlite3
import os
from dotenv import load_dotenv
from typing import TypedDict, List, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

# Google STT / TTS
from google.cloud import speech, texttospeech
import sounddevice as sd
import numpy as np

# ------------------------
# ENV + DB
# ------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DB_FILE = "artisan.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # artisan details
    c.execute("""
        CREATE TABLE IF NOT EXISTS artisans (
            name TEXT PRIMARY KEY,
            craft TEXT,
            experience TEXT,
            location TEXT
        )
    """)
    # chat history
    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            artisan_name TEXT,
            role TEXT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

def add_demo_artisan():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    artisan = ("Ayush", "Artist", "5 years", "Mumbai, India")
    c.execute("INSERT OR REPLACE INTO artisans VALUES (?, ?, ?, ?)", artisan)
    conn.commit()
    conn.close()

# ------------------------
# FETCH DATA
# ------------------------
def get_artisan_details(name):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM artisans WHERE name=?", (name,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "name": row[0],
            "craft": row[1],
            "experience": row[2],
            "location": row[3],
        }
    return None

def get_chat_history(name):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM chats WHERE artisan_name=? ORDER BY id ASC", (name,))
    rows = c.fetchall()
    conn.close()
    messages = []
    for role, content in rows:
        if role == "human":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages

def save_message(name, role, content):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO chats (artisan_name, role, content) VALUES (?, ?, ?)", (name, role, content))
    conn.commit()
    conn.close()

# ------------------------
# LLM + PROMPT
# ------------------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def make_prompt(artisan):
    return ChatPromptTemplate.from_messages([
        ("system", f"You are a master mentor for artisan {artisan['name']}.\n"
                   f"They specialize in {artisan['craft']} with {artisan['experience']} experience "
                   f"from {artisan['location']}. Guide them with useful, clear, and practical advice."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

# ------------------------
# LANGGRAPH
# ------------------------
class State(TypedDict):
    artisan_name: str
    history: List[Any]
    input: str
    output: str

def chat_node(state: State):
    artisan = get_artisan_details(state["artisan_name"])
    history = get_chat_history(state["artisan_name"])
    prompt = make_prompt(artisan)
    chain = prompt | llm
    result = chain.invoke({"history": history, "input": state["input"]})
    save_message(state["artisan_name"], "human", state["input"])
    save_message(state["artisan_name"], "ai", result.content)
    return {
        "output": result.content,
        "history": history + [
            HumanMessage(content=state["input"]),
            AIMessage(content=result.content)
        ]
    }

graph = StateGraph(State)
graph.add_node("chat", chat_node)
graph.set_entry_point("chat")
graph.set_finish_point("chat")

memory = SqliteSaver.from_conn_string("memory.db")
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ------------------------
# GOOGLE STT & TTS HELPERS
# ------------------------
def transcribe_speech() -> str:
    """Record speech from mic and convert to text"""
    client = speech.SpeechClient()
    duration = 5  # seconds
    fs = 16000

    print("🎤 Speak now...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    audio_content = recording.tobytes()

    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=fs,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)
    transcript = response.results[0].alternatives[0].transcript if response.results else ""
    print(f"📝 You said: {transcript}")
    return transcript

def speak_text(text: str):
    """Convert text to speech and play"""
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    audio_data = np.frombuffer(response.audio_content, dtype=np.int16)
    sd.play(audio_data, samplerate=16000)
    sd.wait()

# ------------------------
# MAIN LOOP
# ------------------------
if __name__ == "__main__":
    init_db()
    add_demo_artisan()

    artisan_name = "Ayush"
    thread_id = f"thread_{artisan_name}"
    print(f"🎨 Chat with Artisan Mentor for {artisan_name}\nSay 'exit' anytime to quit.\n")

    history = get_chat_history(artisan_name)
    while True:
        # 🎤 speech-to-text
        user_input = transcribe_speech()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break

        # build state
        state = {
            "artisan_name": artisan_name,
            "history": history,
            "input": user_input,
            "output": ""
        }

        result = app.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}}
        )

        bot_reply = result["output"]
        print("🤖 AI:", bot_reply)

        # 🔊 speak back
        speak_text(bot_reply)

        history = result["history"]
