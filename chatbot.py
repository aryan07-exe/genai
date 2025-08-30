import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import MemorySaver

# ------------------------
# DB SETUP
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
    artisan = ("John", "Artist", "3 years", "delhi, India")
    c.execute("INSERT OR REPLACE INTO artisans VALUES (?, ?, ?, ?)", artisan)
    conn.commit()
    conn.close()

 
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
from typing import TypedDict, List, Any

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

# graph
graph = StateGraph(State)
graph.add_node("chat", chat_node)
graph.set_entry_point("chat")
graph.set_finish_point("chat")

# attach checkpointer
memory = SqliteSaver.from_conn_string("memory.db") 
checkpointer = MemorySaver() # persistent instead of :memory:
app = graph.compile(checkpointer=checkpointer)

# ------------------------
# MAIN LOOP
# ------------------------
if __name__ == "__main__":
    init_db()
    add_demo_artisan()

    artisan_name = "John"
    thread_id = f"thread_{artisan_name}"  # 👈 unique per artisan
    print(f"Chat with Artisan Mentor for {artisan_name}\n")

    history = get_chat_history(artisan_name)
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        state = {
            "artisan_name": artisan_name,
            "history": history,
            "input": user_input,
            "output": ""
        }

        # FIX: pass thread_id via config
        result = app.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}}
        )

        print("AI:", result["output"])
        history = result["history"]
