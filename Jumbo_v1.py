import streamlit as st
import os
import warnings
import logging
from typing import List, Tuple, Optional, Dict, Any
import random
import json
import base64
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process
import chromadb
from chromadb.config import Settings
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Keep warnings quiet
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration management"""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    DEFAULT_MODEL = "groq/llama-3.1-8b-instant"
    MEMORY_DIR = "./jumbo_memory"
    MAX_MEMORIES_PER_QUERY = 5
    MEMORY_CLEANUP_DAYS = 30
    MIN_RELEVANCE_THRESHOLD = 0.6
    MAX_MESSAGE_HISTORY = 50
    DEFAULT_ASSETS_DIR = "./assets"
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    CACHE_TTL = 300  # 5 minutes

MOOD_KEYWORDS = {
    "happy": ["happy", "good", "great", "wonderful", "amazing", "fantastic", "excellent", "joy", "joyful", "elated", "thrilled", "overjoyed", "blessed", "grateful", "content", "cheerful"],
    "excited": ["excited", "pumped", "energetic", "enthusiastic", "awesome", "brilliant", "stoked", "hyped", "fired up", "motivated"],
    "sad": ["sad", "upset", "depressed", "hurt", "pain", "down", "devastated", "heartbroken", "grief", "melancholy", "blue", "disappointed"],
    "anxious": ["anxious", "worried", "scared", "nervous", "stressed", "overwhelmed", "panicked", "stress", "stressful", "tense", "restless", "uneasy"],
    "angry": ["angry", "frustrated", "mad", "annoyed", "irritated", "furious", "pissed", "rage", "outraged", "livid", "resentful"],
    "lost": ["lost", "confused", "uncertain", "unsure", "stuck", "directionless", "bewildered", "perplexed", "aimless"],
    "work_stress": ["work", "job", "career", "office", "boss", "colleague", "deadline", "workload", "promotion", "interview", "meeting", "project", "role", "transition", "burnout"],
    "lonely": ["lonely", "alone", "isolated", "disconnected", "abandoned", "solitary", "empty"],
    "proud": ["proud", "accomplished", "achieved", "successful", "victorious", "triumphant"]
}

JUMBO_RESPONSES = {
    "happy": [
        "That sounds really wonderful! You deserve to feel this good.",
        "I can hear the happiness in your words. It's beautiful to see you shine.",
        "You sound so content right now. That warmth is contagious.",
        "What a lovely thing to share. You seem genuinely happy.",
        "Your joy is radiating through your words. This is beautiful to witness."
    ],
    "excited": [
        "Your excitement is absolutely infectious! You sound ready to take on the world.",
        "I can feel your energy through your words. That enthusiasm is amazing.",
        "You sound pumped up and ready for whatever's coming. That's awesome.",
        "Your passion is lighting up our conversation. Keep that fire burning!"
    ],
    "sad": [
        "It sounds like you're going through something really tough right now.",
        "You seem to be carrying some heavy feelings. That must be hard.",
        "I can hear the sadness in your words. You don't have to go through this alone.",
        "It sounds like things feel pretty overwhelming right now.",
        "Your pain is valid and real. I'm here to sit with you through this."
    ],
    "anxious": [
        "It sounds like your mind is racing with worry right now.",
        "You seem really unsettled about this. That anxiety must be exhausting.",
        "I can hear how much this is weighing on you. You don't have to face it alone.",
        "It sounds like you're feeling pretty overwhelmed. That's completely understandable.",
        "Your worries make complete sense given what you're facing."
    ],
    "angry": [
        "You sound really frustrated about this. That anger makes complete sense.",
        "It sounds like something really got under your skin. You have every right to feel upset.",
        "I can hear how irritated you are. Those feelings are totally valid.",
        "Your frustration is completely justified. Anyone would feel this way."
    ],
    "lost": [
        "You seem unsure about where things are heading, and that can be really unsettling.",
        "It sounds like you're in a confusing place right now. That uncertainty is tough.",
        "You seem to be searching for direction. It's okay not to have all the answers.",
        "Feeling lost is part of the human experience. You're not alone in this uncertainty."
    ],
    "work_stress": [
        "Work stress can be incredibly draining. You're handling a lot right now.",
        "Career transitions and workplace pressure are genuinely challenging.",
        "It sounds like your professional life is weighing heavily on you right now.",
        "Work-life balance is tough, and you're navigating some real challenges."
    ],
    "lonely": [
        "Loneliness can feel so heavy. You're reaching out, and that takes courage.",
        "Feeling disconnected is one of the hardest human experiences.",
        "You're not as alone as you feel right now. I'm here with you.",
        "Isolation is painful, but you're taking steps to connect by being here."
    ],
    "proud": [
        "You should absolutely celebrate this achievement! You've earned this feeling.",
        "Your sense of accomplishment comes through so clearly. Well done!",
        "It's wonderful to hear you recognizing your own success.",
        "You sound genuinely proud, and that pride is well-deserved."
    ],
    "neutral": [
        "I hear you. Thanks for sharing that with me.",
        "You're on my mind. What you're going through matters.",
        "I'm glad you felt comfortable sharing that.",
        "You don't have to carry this alone."
    ],
    "trust": [
        "I'm Jumbo, and you can absolutely trust me as your friend. This conversation is completely private between us.",
        "You can trust me, Jumbo, completely. No one else has access to what you share with me - it's just between friends.",
        "I'm Jumbo, your friend, and this space is totally safe. What you tell me stays between us, always.",
        "You can feel safe opening up to me - I'm Jumbo, and our conversations are private and secure."
    ]
}

def enhanced_mood_detection(text: str) -> Tuple[str, float]:
    """Enhanced mood detection using both keywords and sentiment analysis"""
    try:
        text_lower = text.lower()
        trust_keywords = ["trust", "safe", "safety", "secure", "private", "confidential", "remember", "friend"]
        if any(keyword in text_lower for keyword in trust_keywords):
            return "trust", 0.9
        mood_scores = {}
        for mood, keywords in MOOD_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                mood_scores[mood] = score / len(keywords)
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            if polarity > 0.3:
                sentiment_mood = "happy"
                sentiment_confidence = min(polarity + 0.5, 1.0)
            elif polarity < -0.3:
                sentiment_mood = "sad"
                sentiment_confidence = min(abs(polarity) + 0.5, 1.0)
            elif subjectivity > 0.7:
                sentiment_mood = "anxious"
                sentiment_confidence = subjectivity
            else:
                sentiment_mood = "neutral"
                sentiment_confidence = 0.5
            if mood_scores:
                top_keyword_mood = max(mood_scores.items(), key=lambda x: x[1])
                if top_keyword_mood[1] > sentiment_confidence:
                    return top_keyword_mood, top_keyword_mood[1]
                else:
                    return sentiment_mood, sentiment_confidence
            else:
                return sentiment_mood, sentiment_confidence
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            if mood_scores:
                top_mood = max(mood_scores.items(), key=lambda x: x[1])
                return top_mood, top_mood[1]
        return "neutral", 0.5
    except Exception as e:
        logger.error(f"Mood detection failed: {e}")
        return "neutral", 0.5

def extract_name_from_text(text: str) -> Optional[str]:
    """Enhanced name extraction with better patterns"""
    text_clean = text.lower().strip()
    patterns = [
        (r"my name is ([a-zA-Z]+)", 1),
        (r"i'm ([a-zA-Z]+)", 1),
        (r"i am ([a-zA-Z]+)", 1),
        (r"call me ([a-zA-Z]+)", 1),
        (r"name's ([a-zA-Z]+)", 1),
        (r"i go by ([a-zA-Z]+)", 1)
    ]
    import re
    for pattern, group in patterns:
        match = re.search(pattern, text_clean)
        if match:
            name = match.group(group).strip()
            if len(name) > 1 and name.isalpha():
                return name.title()
    return None

def make_llm(groq_api_key: str, model: str = "groq/llama-3.1-8b-instant") -> ChatGroq:
    if not groq_api_key:
        raise ValueError("No Groq API key found.")
    return ChatGroq(
        groq_api_key=groq_api_key.strip(),
        model=model,
        temperature=0.7,
        max_tokens=500,
        timeout=None
    )

class EnhancedJumboMemory:
    """Enhanced memory system with better data management and analytics"""
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.collection_name = f"jumbo_memory_{hashlib.md5(user_id.encode()).hexdigest()}"
        try:
            os.makedirs(Config.MEMORY_DIR, exist_ok=True)
            self.client = chromadb.Client(Settings(
                persist_directory=Config.MEMORY_DIR,
                is_persistent=True
            ))
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                self.collection = self.client.get_collection(name=self.collection_name)
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            raise

    def store_conversation(self, user_message: str, jumbo_response: str, mood: str, 
                          confidence: float, user_name: str = None) -> bool:
        """Store conversation with enhanced metadata"""
        try:
            conversation_id = hashlib.md5(
                f"{datetime.now().isoformat()}{user_message}".encode()
            ).hexdigest()
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "mood": mood,
                "confidence": confidence,
                "type": "conversation",
                "message_length": len(user_message),
                "response_length": len(jumbo_response)
            }
            if user_name:
                metadata["user_name"] = user_name
            self.collection.add(
                documents=[f"User: {user_message}\nJumbo: {jumbo_response}"],
                metadatas=[metadata],
                ids=[conversation_id]
            )
            # Cleanup old conversations
            self._cleanup_old_memories()
            return True
        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")
            return False

    def store_user_info(self, info_type: str, info_value: str):
        """Store specific user information like name"""
        info_id = f"{self.user_id}_{info_type}"
        try:
            existing = self.collection.get(ids=[info_id])
            if existing['ids']:
                self.collection.update(
                    ids=[info_id],
                    documents=[f"{info_type}: {info_value}"],
                    metadatas=[{
                        "timestamp": datetime.now().isoformat(),
                        "type": "user_info",
                        "info_type": info_type
                    }]
                )
            else:
                raise Exception("Not found")
        except Exception:
            self.collection.add(
                documents=[f"{info_type}: {info_value}"],
                metadatas=[{
                    "timestamp": datetime.now().isoformat(),
                    "type": "user_info",
                    "info_type": info_type
                }],
                ids=[info_id]
            )

    def get_user_name(self) -> Optional[str]:
        """Retrieve user's name if stored"""
        try:
            info_id = f"{self.user_id}_name"
            result = self.collection.get(ids=[info_id])
            if result['ids']:
                document = result['documents']
                return document.split(": ")[1] if ": " in document else None
        except Exception:
            pass
        return None

    def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant past conversations"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where={"type": "conversation"}
            )
            memories = []
            if results['documents'] and results['documents']:
                for i, doc in enumerate(results['documents']):
                    memories.append({
                        "content": doc,
                        "metadata": results['metadatas'][i] if results['metadatas'] and results['metadatas'] else {},
                        "distance": results['distances'][i] if results['distances'] and results['distances'] else 1.0
                    })
            return memories
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return []

class EnhancedJumboCrew:
    """Enhanced emotional wellness chatbot with improved error handling and caching"""
    def __init__(self, groq_api_key: Optional[str] = None, user_id: str = "default_user"):
        self.user_id = user_id
        self.api_key = groq_api_key or Config.GROQ_API_KEY
        if not self.api_key:
            raise ValueError("No Groq API key found.")
        try:
            self.llm = make_llm(self.api_key)
            self.memory = EnhancedJumboMemory(user_id)
            self.listener, self.companion, self.summariser = self._create_agents()
            self._cached_crew = None
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedJumboCrew: {e}")
            raise

    def _create_agents(self) -> Tuple[Agent, Agent, Agent]:
        """Create enhanced agents with better error handling"""
        try:
            listener_agent = Agent(
                role="Advanced Emotion Detector",
                goal="Identify emotional states, context, and support needs with high accuracy, considering conversation history and user patterns.",
                backstory="""You are an expert in emotional intelligence and human psychology. You can detect subtle emotional cues, understand context from past conversations, and identify what kind of support would be most helpful. You consider both explicit statements and implicit feelings, analyzing mood patterns over time.""",
                verbose=False,
                llm=self.llm
            )
            companion_agent = Agent(
                role="Jumbo the Wise Elephant Companion", 
                goal="Provide empathetic, personalized responses that make users feel heard, understood, and supported while maintaining conversation continuity.",
                backstory="""You are Jumbo, a wise and caring elephant with an excellent memory and deep emotional intelligence. You remember past conversations and use this knowledge to provide personalized, continuous support.

Your core principles:
- Always use "you" language to reflect the user's feelings
- Never impose your own feelings with "I feel" statements
- Remember and reference past conversations naturally
- Introduce yourself as "Jumbo" when discussing trust/safety
- Provide specific, relevant responses (not generic emotional validation)
- Match the emotional tone appropriately
- Use the user's name when you know it
- Be warm, conversational, and genuine
- Ask thoughtful follow-up questions
- Maintain user privacy and confidentiality

You have perfect memory of past conversations and can reference them to provide continuity and show that you truly care about the user's journey.""",
                verbose=False,
                llm=self.llm
            )
            summariser_agent = Agent(
                role="Response Quality Enhancer",
                goal="Ensure responses are perfectly crafted - empathetic, conversational, appropriately emotional, and end with thoughtful questions.",
                backstory="""You are responsible for ensuring every response meets the highest standards of emotional support. You refine responses to be natural, warm, and conversational while maintaining therapeutic value. You ensure proper "you" language, appropriate emotional matching, and meaningful follow-up questions.""",
                verbose=False,
                llm=self.llm
            )
            return listener_agent, companion_agent, summariser_agent
        except Exception as e:
            logger.error(f"Failed to create agents: {e}")
            raise

    def respond(self, user_message: str, max_retries: int = Config.MAX_RETRIES) -> Tuple[str, Dict]:
        """Generate response with enhanced error handling and retries"""
        response_metadata = {
            "timestamp": datetime.now().isoformat(),
            "processing_time": 0,
            "mood_detected": "neutral",
            "confidence": 0.5,
            "memories_used": 0,
            "success": False,
            "error": None
        }
        start_time = time.time()
        try:
            extracted_name = extract_name_from_text(user_message)
            if extracted_name:
                self.memory.store_user_info("name", extracted_name)
            user_name = self.memory.get_user_name()
            detected_mood, confidence = enhanced_mood_detection(user_message)
            response_metadata.update({
                "mood_detected": detected_mood,
                "confidence": confidence
            })
            relevant_memories = self.memory.get_relevant_memories(user_message, limit=3)
            response_metadata["memories_used"] = len(relevant_memories)
            memory_context = ""
            if relevant_memories:
                memory_context = "\n\nRelevant past conversations:\n"
                for memory in relevant_memories[:2]:
                    memory_context += f"- {memory['content'][:200]}...\n"
            for attempt in range(max_retries):
                try:
                    response = self._generate_response(
                        user_message, user_name, detected_mood, 
                        confidence, memory_context, relevant_memories
                    )
                    if response and len(response.strip()) >= 10:
                        self.memory.store_conversation(
                            user_message, response, detected_mood, confidence, user_name
                        )
                        response_metadata.update({
                            "processing_time": time.time() - start_time,
                            "success": True
                        })
                        return response, response_metadata
                except Exception as e:
                    logger.warning(f"Response generation attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            response_metadata["error"] = str(e)
            response = self._get_fallback_response(detected_mood, user_name)
            self.memory.store_conversation(
                user_message, response, detected_mood, confidence, user_name
            )
            response_metadata.update({
                "processing_time": time.time() - start_time,
                "success": False
            })
            return response, response_metadata

    def _generate_response(self, user_message: str, user_name: Optional[str], 
                          mood: str, confidence: float, memory_context: str,
                          relevant_memories: List[Dict]) -> str:
        """Generate response using CrewAI"""
        listen_task = Task(
            description=f"""Analyze the emotional content and context of this message: '{user_message}'

            User Information:
            - Name: {user_name or 'Unknown'}
            - Detected mood: {mood} (confidence: {confidence:.2f})
            - Available memory context: {len(relevant_memories)} relevant past conversations

            {memory_context}

            Provide a comprehensive emotional analysis that considers:
            1. Current emotional state and intensity
            2. Underlying needs or concerns
            3. Connection to past conversations (if any)
            4. What type of support would be most helpful
            5. Any patterns or themes you notice""",
            agent=self.listener,
            expected_output="Detailed emotional analysis with context and support recommendations"
        )
        companion_task = Task(
            description=f"""Create a warm, empathetic response as Jumbo the elephant. Use these STRICT GUIDELINES:

            Message to respond to: '{user_message}'

            Context:
            - User's name: {user_name or 'Unknown'}
            - Mood: {mood} (confidence: {confidence:.2f})
            - Memory context available: {'Yes' if relevant_memories else 'No'}

            {memory_context}

            RESPONSE REQUIREMENTS:
            1. Always use "you" language ("You sound...", "You seem...", "That must be...")
            2. NEVER use "I feel" or "I think you..." constructions
            3. Address the SPECIFIC content they shared (not generic emotional validation)
            4. Use their name ({user_name}) naturally if known
            5. Reference past conversations when relevant and supportive
            6. Match the emotional tone to their {mood} state
            7. For trust/safety questions: introduce yourself as "Jumbo"
            8. Keep response conversational and natural (2-3 sentences)
            9. Be specific and relevant to what they actually said
            10. Show genuine understanding and care

            Create a response that directly addresses their message content while being emotionally supportive.""",
            agent=self.companion,
            context=[listen_task],
            expected_output="Personalized, empathetic response from Jumbo addressing specific user content"
        )
        summariser_task = Task(
            description=f"""Refine the response to ensure it meets all quality standards:

            Requirements:
            1. Uses "you" language throughout (never "I feel")
            2. Sounds natural and conversational (not clinical)
            3. Matches the {mood} mood appropriately
            4. Incorporates user's name ({user_name}) naturally if known
            5. References memories when relevant and helpful
            6. Ends with a thoughtful, gentle follow-up question
            7. Feels warm, supportive, and judgment-free
            8. Is concise (2-3 sentences) but emotionally rich
            9. Shows Jumbo's caring personality
            10. Addresses the specific content they shared

            Ensure the final response would make the user feel truly heard and understood.""",
            agent=self.summariser,
            context=[companion_task],
            expected_output="Perfectly crafted empathetic response from Jumbo (2-3 sentences with gentle question)"
        )
        crew = Crew(
            agents=[self.listener, self.companion, self.summariser],
            tasks=[listen_task, companion_task, summariser_task],
            verbose=False,
            process=Process.sequential
        )
        result = crew.kickoff()
        return str(result).strip()

    def _get_fallback_response(self, mood: str, user_name: Optional[str]) -> str:
        """Generate mood-appropriate fallback responses"""
        name_part = f", {user_name}" if user_name else ""
        fallback_responses = {
            "happy": [
                f"That sounds wonderful{name_part}! You seem really content right now. What's been bringing you the most joy lately?",
                f"Your happiness is shining through{name_part}! It's beautiful to witness. What's making this such a good time for you?"
            ],
            "sad": [
                f"You sound like you're going through something difficult{name_part}. Those heavy feelings are completely valid. What's been weighing on your heart?",
                f"I can hear the sadness in your words{name_part}. You don't have to go through this alone. What's been the hardest part?"
            ],
            "anxious": [
                f"It sounds like your mind has been racing with worry{name_part}. That anxiety must be exhausting. What's been making you feel most unsettled?",
                f"You seem really overwhelmed right now{name_part}. Those anxious feelings make complete sense. What's weighing most heavily on you?"
            ],
            "angry": [
                f"You sound really frustrated{name_part}, and that anger makes complete sense. Those feelings are totally valid. What's been getting under your skin?",
                f"I can hear how irritated you are{name_part}. Your frustration is completely justified. What's been pushing your buttons?"
            ],
            "excited": [
                f"Your excitement is contagious{name_part}! You sound ready to take on the world. What's got you feeling so energized?",
                f"I can feel your enthusiasm through your words{name_part}! That energy is amazing. What's lighting you up right now?"
            ],
            "lost": [
                f"You seem to be in a confusing place right now{name_part}. That uncertainty can be really tough. What's been making you feel most unsure?",
                f"It sounds like you're searching for direction{name_part}. That's completely understandable. What's been on your mind about your path forward?"
            ],
            "work_stress": [
                f"Work stress can be incredibly draining{name_part}. You're handling a lot right now. What's been the most challenging part of your work situation?",
                f"Career pressure is genuinely tough{name_part}. You're navigating some real challenges. What aspect of work has been weighing on you most?"
            ],
            "lonely": [
                f"Loneliness can feel so heavy{name_part}. You're reaching out, and that takes courage. What's been making you feel most disconnected?",
                f"Feeling isolated is one of the hardest experiences{name_part}. You're not as alone as you feel. What's been contributing to this loneliness?"
            ],
            "proud": [
                f"You should absolutely celebrate this{name_part}! You've earned this feeling of accomplishment. What achievement are you most proud of?",
                f"Your sense of pride comes through so clearly{name_part}! Well done. What success are you celebrating?"
            ],
            "trust": [
                f"I'm Jumbo{name_part}, and you can absolutely trust me as your friend. This conversation is completely private between us. What's been weighing on your heart lately?",
                f"You can trust me completely{name_part}. I'm Jumbo, and our conversations are private and secure. What would you like to share?"
            ]
        }
        responses = fallback_responses.get(mood, [
            f"I hear you{name_part}, and I want you to know that your feelings are valid. What's been on your mind lately?",
            f"Thank you for sharing that with me{name_part}. Your thoughts and feelings matter. What's been weighing on you?"
        ])
        return random.choice(responses) + " 🐘💙"

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #E8F4FD 0%, #B8E0F5 100%);
    }

    .gif-banner {
        width: 100%;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }

    .chat-message {
        background: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    .user-message {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        text-align: right;
    }

    .jumbo-message {
        background: linear-gradient(135deg, #a8e6cf 0%, #7fcdcd 100%);
        color: #2c3e50;
    }

    .welcome-container {
        text-align: center;
        background: rgba(255, 255, 255, 0.9);
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }

    .elephant-emoji {
        font-size: 100px;
        margin-bottom: 20px;
    }

    .input-section {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    .memory-info {
        background: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Jumbo - Your Emotional Assistant with Memory",
    page_icon="🐘",
    layout="centered"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "crew" not in st.session_state:
    st.session_state.crew = None
if "started" not in st.session_state:
    st.session_state.started = False
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{hash(str(st.session_state)) % 10000}"

if st.session_state.crew is None:
    api_key = os.getenv("GROQ_API_KEY") or "your_api_key_here"
    if api_key and api_key != "your_api_key_here":
        try:
            st.session_state.crew = EnhancedJumboCrew(groq_api_key=api_key, user_id=st.session_state.user_id)
        except Exception as e:
            st.error(f"Error initializing Jumbo: {e}")
            st.info("Please set your GROQ_API_KEY environment variable or add chromadb: `pip install chromadb`")
            st.stop()
    else:
        st.error("Please set your GROQ_API_KEY environment variable")
        st.stop()

st.markdown('<div class="main-container">', unsafe_allow_html=True)

gif_path = r"D:\MOOD\CODE\images\Title.gif"
if os.path.exists(gif_path):
    with open(gif_path, "rb") as file:
        gif_data = base64.b64encode(file.read()).decode()
        st.markdown(f"""
        <div style="text-align: center; margin: -20px -20px 10px -20px; height: 200px; overflow: hidden; border-radius: 10px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); position: relative;">
            <img src="data:image/gif;base64,{gif_data}" 
                 style="width: 100%; height: auto; display: block; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);" />
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #a8e6cf 0%, #7fcdcd 100%); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: #2c3e50; font-size: 3rem;">🐘 Jumbo</h1>
        <p style="color: #2c3e50; font-size: 1.5rem;">Your Emotional Assistant with Memory</p>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.crew:
    with st.sidebar:
        st.markdown("""
        <div class="memory-info">
            <h3 style="color: #2c3e50;">🧠 Memory Status</h3>
        </div>
        """, unsafe_allow_html=True)

        user_name = st.session_state.crew.memory.get_user_name()
        if user_name:
            st.success(f"Jumbo remembers: **{user_name}** 😊")
        else:
            st.info("Jumbo doesn't know your name yet. Try introducing yourself!")

        if st.button("🗑️ Clear Memory"):
            try:
                collection_name = f"jumbo_memory_{st.session_state.user_id}"
                st.session_state.crew.memory.client.delete_collection(collection_name)
                st.success("Memory cleared! 🧹")
                st.rerun()
            except Exception:
                st.error("Error clearing memory")

if not st.session_state.started:
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            gif_path = r"D:\MOOD\CODE\images\elephant.gif"
            with open(gif_path, "rb") as f:
                gif_bytes = f.read()
                b64_gif = base64.b64encode(gif_bytes).decode()

            st.markdown(
                f'<div style="text-align: center;">'
                f'<img src="data:image/gif;base64,{b64_gif}" width="400" />'
                f'</div>',
                unsafe_allow_html=True
            )
        with col2:
            user_name = st.session_state.crew.memory.get_user_name() if st.session_state.crew else None
            greeting = f"Hi {user_name}, I'm Jumbo!" if user_name else "Hi, I'm Jumbo!"
            st.markdown(f"""
                <h1 style="color: #2c3e50; margin-bottom: 15px;">{greeting}</h1>
                <p style="color: #2c3e50; font-size: 24px;">
                    I'm here to listen, remember, and support you through whatever you're experiencing. I use advanced emotional intelligence and keep track of our conversations to provide personalized, continuous support.
                </p>
                <h3 style="color: #2c3e50; margin-top: 30px;">How are you feeling today?</h3>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    user_input = st.text_area("Tell me what's on your mind...", 
                             height=100, 
                             placeholder="I'm feeling... / Today I... / I need help with... / My name is...")
    if st.button("Share with Jumbo 🐘"):
        if user_input and user_input.strip():
            st.session_state.started = True
            st.session_state.messages.append({"role": "user", "content": user_input})
            try:
                response, _ = st.session_state.crew.respond(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                mood, _ = enhanced_mood_detection(user_input)
                fallback_response = f"Thank you for sharing that with me. I can sense you're feeling {mood}. I'm here to support you through this. What would help you feel better right now? 🐘💙"
                st.session_state.messages.append({"role": "assistant", "content": fallback_response})
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.started:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message jumbo-message">
                <strong>🐘 Jumbo:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("**Continue the conversation:**")
    new_input = st.text_area("What else would you like to share?", 
                            height=80, 
                            key="continuing_chat")
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Send 💬"):
            if new_input and new_input.strip():
                st.session_state.messages.append({"role": "user", "content": new_input})
                try:
                    response, _ = st.session_state.crew.respond(new_input)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    fallback_responses = [
                        "I hear you, and I want you to know that your feelings are valid. Tell me more about what you're experiencing. 🐘",
                        "Thank you for trusting me with your thoughts. I'm here to support you. What's weighing on your heart? 💙",
                        "I can sense this is important to you. I'm listening with my whole heart. What do you need right now? 🐘"
                    ]
                    response = random.choice(fallback_responses)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    with col2:
        if st.button("New Topic 🔄"):
            user_name = st.session_state.crew.memory.get_user_name() if st.session_state.crew else None
            name_part = f" {user_name}" if user_name else ""
            response = f"What else is on your mind{name_part}? I'm here to listen to whatever you'd like to share. 🐘💙"
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #2c3e50; opacity: 0.7; padding: 10px;'>
    🐘 <em>Jumbo is here for you - Your feelings are always valid and remembered</em> 💙
</div>
""", unsafe_allow_html=True)
