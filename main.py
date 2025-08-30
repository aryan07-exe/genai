import sqlite3
import os
from typing import TypedDict, List, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# ------------------------
# ENV + DB INIT
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

init_db()

# ------------------------
# DB HELPERS
# ------------------------
def add_artisan(name: str, craft: str, experience: str, location: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO artisans VALUES (?, ?, ?, ?)",
              (name, craft, experience, location))
    conn.commit()
    conn.close()

def get_artisan_details(name: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM artisans WHERE name=?", (name,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"name": row[0], "craft": row[1], "experience": row[2], "location": row[3]}
    return None

def get_chat_history(name: str):
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

def save_message(name: str, role: str, content: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO chats (artisan_name, role, content) VALUES (?, ?, ?)", (name, role, content))
    conn.commit()
    conn.close()

# ------------------------
# LLM + PROMPT
# ------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

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

checkpointer = MemorySaver()
app_graph = graph.compile(checkpointer=checkpointer)

# ------------------------
# FASTAPI APP
# ------------------------
app = FastAPI()

# -------- Request Models --------
class RegisterRequest(BaseModel):
    name: str
    craft: str
    experience: str
    location: str

class ChatRequest(BaseModel):
    username: str
    message: str

# -------- API Endpoints --------
@app.post("/register")
def register(req: RegisterRequest):
    if get_artisan_details(req.name):
        raise HTTPException(status_code=400, detail="User already exists")
    add_artisan(req.name, req.craft, req.experience, req.location)
    return {"message": f"Artisan {req.name} registered successfully"}

@app.get("/login/{username}")
def login(username: str):
    artisan = get_artisan_details(username)
    if not artisan:
        raise HTTPException(status_code=404, detail="User not found, please register")
    return {"message": f"Welcome back {username}", "artisan": artisan}

@app.post("/chat")
def chat(req: ChatRequest):
    artisan = get_artisan_details(req.username)
    if not artisan:
        raise HTTPException(status_code=404, detail="User not found, please register first")

    history = get_chat_history(req.username)
    state = {
        "artisan_name": req.username,
        "history": history,
        "input": req.message,
        "output": ""
    }

    thread_id = f"thread_{req.username}"
    result = app_graph.invoke(state, config={"configurable": {"thread_id": thread_id}})

    return {"response": result["output"]}
