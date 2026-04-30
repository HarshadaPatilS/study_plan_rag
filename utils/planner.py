"""
planner.py
Core RAG pipeline: retrieves relevant syllabus content, then uses
Groq LLM (free, fast) to generate a personalized study plan.
"""

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from utils.vectorstore import retrieve_relevant_chunks, FAISS


SYSTEM_PROMPT = """You are an expert academic coach and study planner.
Your job is to create structured, realistic, day-by-day study plans based on the student's actual syllabus content.

Rules:
- Be specific — reference actual topics from the syllabus content provided
- Be realistic — don't overload any single day
- Include revision days and breaks
- Use a clear format: Day 1, Day 2, etc.
- End with an exam-day tip
- Keep motivational tone but stay practical
"""


def build_rag_query(days: int, hours_per_day: float, weak_topics: str, exam_date_note: str) -> str:
    """Build the retrieval query that finds the most relevant syllabus sections."""
    query = f"topics chapters units study plan {days} days"
    if weak_topics:
        query += f" focus on {weak_topics}"
    return query


def generate_study_plan(
    vectorstore: FAISS,
    groq_api_key: str,
    days: int,
    hours_per_day: float,
    weak_topics: str,
    exam_note: str,
    difficulty: str,
) -> str:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from FAISS
    2. Inject into prompt as context
    3. Call Groq LLM to generate the plan
    """

    # Step 1: Retrieve relevant syllabus content
    query = build_rag_query(days, hours_per_day, weak_topics, exam_note)
    context = retrieve_relevant_chunks(vectorstore, query, k=10)

    # Step 2: Build the user prompt
    weak_section = f"\nTopics I find difficult or want extra focus on: {weak_topics}" if weak_topics.strip() else ""
    exam_section = f"\nAdditional context: {exam_note}" if exam_note.strip() else ""

    user_prompt = f"""
Here is the syllabus/notes content extracted from the student's document:

====== SYLLABUS CONTENT ======
{context}
==============================

Based on this syllabus content, create a detailed day-by-day study plan with the following constraints:

- Total study days available: {days} days
- Study hours per day: {hours_per_day} hours
- Difficulty/intensity preference: {difficulty}{weak_section}{exam_section}

Create a personalized study schedule that:
1. Covers all major topics from the syllabus
2. Allocates more time to complex topics
3. Includes 1 revision day for every 4-5 study days
4. Suggests specific resources or practice activities per topic
5. Ends with a final day exam strategy

Format each day as:
**Day X — [Topic Name]**
- Time: X hours
- Goal: ...
- Tasks: ...
- Quick tip: ...
"""

    # Step 3: Call Groq LLM (Llama 3.1 70B — free tier)
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-70b-versatile",
        temperature=0.4,
        max_tokens=4096,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    return response.content
