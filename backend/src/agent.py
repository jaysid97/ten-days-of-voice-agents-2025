import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Annotated, Optional

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, deepgram, silero

# Load environment variables
load_dotenv(dotenv_path=".env.local")

# Initialize Logger
logger = logging.getLogger("jaysid-sdr")
logger.setLevel(logging.INFO)

# ======================================================
# 📂 1. KNOWLEDGE BASE & CONFIGURATION
# ======================================================

COMPANY_NAME = "Jaysid Development"
FAQ_FILE = "jaysid_faq.json"
LEADS_FILE = "leads_db.json"

# Default Knowledge Base
DEFAULT_FAQ = [
    {
        "question": "What does Jaysid Development do?",
        "answer": "We are a premier software house specializing in Custom AI Agents, Full-Stack Web Development, and Cloud Migration services."
    },
    {
        "question": "How much does a custom AI agent cost?",
        "answer": "Our pilot packages start at $2,500 for a basic RAG chatbot. Enterprise voice agents typically range from $10k to $25k depending on integration complexity."
    },
    {
        "question": "What tech stack do you use?",
        "answer": "We specialize in Python, TypeScript, React, Next.js, and cloud providers like AWS and Google Cloud. For AI, we use OpenAI, Anthropic, and LiveKit."
    },
    {
        "question": "Do you offer staff augmentation?",
        "answer": "Yes, we can provide dedicated senior developers to join your existing team on a contract basis."
    }
]

def load_faq_context() -> str:
    """Loads the FAQ and returns it as a string for the LLM system prompt."""
    if not os.path.exists(FAQ_FILE):
        with open(FAQ_FILE, "w") as f:
            json.dump(DEFAULT_FAQ, f, indent=4)
    
    with open(FAQ_FILE, "r") as f:
        data = json.load(f)
        return json.dumps(data, indent=2)

# ======================================================
# 💾 2. STATE MANAGEMENT & TOOLS
# ======================================================

class LeadManager:
    """Manages the state of the lead during the call."""
    def __init__(self):
        self.profile = {
            "name": None,
            "company": None,
            "email": None,
            "use_case": None,
            "budget": None,
            "timeline": None
        }

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                self.profile[k] = v
        logger.info(f"📝 Profile Updated: {self.profile}")

    def is_complete(self):
        # Minimum requirements to be considered a "Lead"
        return all([self.profile['name'], self.profile['use_case']])

    def save_to_disk(self):
        entry = self.profile.copy()
        entry["timestamp"] = datetime.now().isoformat()
        
        existing_leads = []
        if os.path.exists(LEADS_FILE):
            try:
                with open(LEADS_FILE, "r") as f:
                    existing_leads = json.load(f)
            except json.JSONDecodeError:
                pass
        
        existing_leads.append(entry)
        
        with open(LEADS_FILE, "w") as f:
            json.dump(existing_leads, f, indent=4)
        logger.info(f"💾 Lead saved to {LEADS_FILE}")

# Define the Global State for the specific session
lead_state = LeadManager()

# --- Tool Definitions ---

class SDRTools(llm.FunctionContext):
    
    @llm.ai_callable(description="Update the potential client's profile with new information.")
    def update_lead_profile(
        self,
        name: Annotated[Optional[str], llm.TypeInfo(description="The customer's name")] = None,
        company: Annotated[Optional[str], llm.TypeInfo(description="The customer's company name")] = None,
        email: Annotated[Optional[str], llm.TypeInfo(description="The customer's email address")] = None,
        use_case: Annotated[Optional[str], llm.TypeInfo(description="What software/AI they want to build")] = None,
        budget: Annotated[Optional[str], llm.TypeInfo(description="Their budget range")] = None,
        timeline: Annotated[Optional[str], llm.TypeInfo(description="When they want to start")] = None,
    ):
        lead_state.update(
            name=name, company=company, email=email, 
            use_case=use_case, budget=budget, timeline=timeline
        )
        return "Lead profile updated successfully."

    @llm.ai_callable(description="Save the lead to the database when the conversation is concluding.")
    def submit_lead(self):
        lead_state.save_to_disk()
        return "Lead has been saved to the database."

# ======================================================
# 🧠 3. AGENT ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    
    # 1. Load Knowledge Base
    faq_data = load_faq_context()
    
    # 2. Define Persona & Instructions
    system_prompt = f"""
    You are 'Jay', the AI Sales Representative for '{COMPANY_NAME}'.
    
    **YOUR GOAL:**
    Qualify inbound leads for our software development services. You need to gather information casually while answering their questions.
    
    **KNOWLEDGE BASE:**
    {faq_data}
    
    **REQUIRED INFORMATION (Try to get these):**
    1. Name
    2. Company / Role
    3. What are they trying to build? (Use Case)
    4. Timeline/Budget
    
    **GUIDELINES:**
    - Be professional, enthusiastic, and concise.
    - Do NOT ask for all information at once. Conversation loop: Answer their question -> Ask ONE qualifying question.
    - If you don't know an answer, say you will have a senior engineer email them.
    - When you hear new details (like name or email), call `update_lead_profile` immediately.
    - When the user says goodbye or the conversation ends, call `submit_lead`.
    """

    # 3. Initialize the Voice Pipeline
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-2-general"), # Fast, accurate STT
        llm=openai.LLM(model="gpt-4o"),           # Smartest for function calling
        tts=openai.TTS(voice="alloy"),            # Clean, fast TTS
        fnc_ctx=SDRTools(),                       # Bind our tools
        chat_ctx=llm.ChatContext().append(
            role="system",
            text=system_prompt
        ),
    )

    # 4. Connect and Start
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    print(f"🚀 {COMPANY_NAME} Agent Started in room: {ctx.room.name}")
    
    # Say hello first
    await agent.say("Hi there! Welcome to Jaysid Development. I'm Jay, the AI assistant. Are you looking to build a new software project or upgrade an existing one?", allow_interruptions=True)

    agent.start(ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
