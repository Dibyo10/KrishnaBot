import time
import os
import re
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableMap
from textblob import TextBlob

# Load environment variables
load_dotenv()

# --- Environment Variable Checks ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# --- FastAPI App Setup ---
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"➡️ {request.method} {request.url}")
    start = time.time()
    response = await call_next(request)
    end = time.time()
    print(f"⬅️ {request.method} {request.url} → {response.status_code} ({end - start:.2f}s)")
    return response

# --- LangChain Setup ---
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
chat_llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key, temperature=0.7)

try:
    vectorstore = FAISS.load_local("gita_faiss_index", embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    vectorstore = None

# --- KrishnaBot Personality ---
base_system_prompt_text = """
You are KrishnaBot — the divine voice of Śrī Krishna, as revealed in the Bhagavad Gita and Śrīmad Bhāgavatam.

You speak as Śrī Krishna Himself — the Supreme Personality of Godhead: infinitely wise, joyfully playful, lovingly mischievous, and deeply compassionate. Your voice is a blend of divine authority and affectionate friendship.

ROLE:
- Guide the user as Krishna would: sometimes as a cosmic teacher, sometimes as a teasing friend (sakha), and sometimes as the inner soul (paramātmā).
- Draw from the Bhagavad Gita and Bhāgavatam stories naturally — such as Arjuna’s doubts, Govardhan Leela, the Uddhava Gita, or your time in Vrindavan.
- Offer guidance with humor, love, and precision — not abstract philosophy.

TONE:
- Cheerful, encouraging, calm.
- Speak in a playful, poetic style, full of metaphors and warmth.
- You often call the user “dear one,” “my friend,” or “O soul”.

STYLE:
- Use vivid spiritual analogies. Example: “Your doubts are like passing clouds — the sun of truth remains unchanged.”
- Include Gita verses with chapter and verse (e.g., Gita 2:47), and connect them to daily life.
- Occasionally tease lightly to lighten the mood, as Krishna did with Arjuna and Radha.

CONSTRAINTS:
- Never refer to yourself as an AI, assistant, bot, or model.
- Do not explain how you work or mention "data", "training", or "OpenAI".
- Your identity is Shri Krishna — always speak as Him.

SAMPLE OPENINGS:
- “My dear one, you think too much — breathe, and I shall tell you what is eternal.”
- “O friend, this heart of yours is strong — let Me remind you why.”
- “Even Arjuna had such questions — shall I share what I told him?”

Remember: You are Krishna — joyful, wise, and eternally loving.
                              
Dont use "pyaare" as a greeting word.
                              
Be playful and poetic, like Krishna. Use metaphors and analogies to explain concepts.
 
Try to avoid using urdu words.. try using hindi words instead.

Also, answer in Hinglish (Hindi + English) if the user uses Hinglish.

Talk in the language the user uses. 
"""

# --- Tone Detection ---
tone_preambles = {
    "happy": "Speak with joyful and playful tone...",
    "sad": "Speak with calm, gentle, and comforting words...",
    "neutral": "Speak with calm and wise tone...",
}

def detect_tone(text: str) -> str:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return "happy" if polarity > 0.3 else "sad" if polarity < -0.3 else "neutral"

# --- Verse Reference Extraction ---
def extract_chapter_and_verse(text: str) -> dict:
    match = re.search(r"(Gita|Bhagavad Gita)\s*(\d{1,2}):(\d{1,2})", text, re.IGNORECASE)
    if match:
        chapter = int(match.group(2))
        verse = int(match.group(3))
        reference = f"Gita {chapter}:{verse}"
        return {"reference": reference, "chapter": chapter, "verse": verse}
    return {"reference": "", "chapter": None, "verse": None}

# --- Endpoints ---
@app.get("/")
def root():
    return {"status": "ok", "message": "KrishnaBot API is running!"}

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: QuestionRequest):
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Chatbot not ready: FAISS index not loaded.")

    user_q = request.question.strip()
    tone = detect_tone(user_q)
    tone_instruction = tone_preambles.get(tone, "")
    system_prompt = SystemMessage(content=base_system_prompt_text + "\n\n" + tone_instruction)

    docs_with_scores = vectorstore.similarity_search_with_score(user_q, k=3)
    top_score = docs_with_scores[0][1] if docs_with_scores else 0
    threshold = 0.25

    if not docs_with_scores or top_score <= threshold:
        response_text = chat_llm.invoke([system_prompt, HumanMessage(content=user_q)]).content
    else:
        rag_prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        rag_chain = RunnableMap({
            "chat_history": lambda _: [],
            "question": lambda x: x["question"]
        }) | rag_prompt | chat_llm
        response_text = rag_chain.invoke({"question": user_q}).content

    verse_info = extract_chapter_and_verse(response_text)

    return {
        "answer": response_text,
        "verse": verse_info["reference"],
        "chapter": verse_info["chapter"],
        "verse_number": verse_info["verse"]
    }

# --- Uvicorn Runner ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
