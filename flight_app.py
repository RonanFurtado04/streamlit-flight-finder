import os
import streamlit as st
from dotenv import load_dotenv
import random
from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_realtime_flight_data():
    """
    Simulates a real-time API by generating a random list of flight details for today.
    """
    airlines = ["IndiGo", "Vistara", "Air India", "Air India Express", "SpiceJet"]
    destinations = {
        "Delhi (DEL)": (6000, 12000),
        "Mumbai (BOM)": (5500, 11000),
        "Dubai (DXB)": (11000, 25000),
        "Singapore (SIN)": (14000, 30000),
        "London (LHR)": (45000, 90000)
    }
    docs = []
    today_str = "25 Sep 2025"
    
    for _ in range(20): 
        airline = random.choice(airlines)
        dest, price_range = random.choice(list(destinations.items()))
        price = random.randint(price_range[0], price_range[1])
        departure_time = (datetime(2025, 9, 25, 5, 0) + timedelta(minutes=random.randint(0, 18 * 60))).strftime("%H:%M")
        flight_num = f"{airline[:2].upper()}-{random.randint(100, 999)}"
        
        doc_content = (
            f"Airline: {airline}. Flight: {flight_num}. Route: Bengaluru (BLR) to {dest}. "
            f"Departure: {today_str}, {departure_time}. Status: Available. "
            f"Price: {price:,} INR."
        )
        docs.append(Document(page_content=doc_content))
        
    return docs

def get_rag_chain(docs):
    """
    This function sets up the RAG pipeline with the provided documents.
    """
 
    load_dotenv()
    
   
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    
 
    vector = FAISS.from_documents(docs, embeddings)
    
   
    prompt = ChatPromptTemplate.from_template("""
    You are a real-time flight price assistant. Your goal is to find the cheapest flight options for the user for today.
    Answer the following question based ONLY on the provided real-time flight details.
    When asked for the cheapest flight, identify the flight with the lowest price from the list. Provide the airline, destination, and price.
    If the information is not in the context, say "I could not find a flight for that query in the current real-time data."

    <context>
    {context}
    </context>

    Question: {input}
    """)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain



st.set_page_config(page_title="Real-Time Flight Finder", layout="wide")
st.title("✈️ Real-Time Cheapest Flight Finder")
st.write(f"Finding the best deals from Bengaluru (BLR) for **Today, September 25, 2025**.")


st.sidebar.title("About")
st.sidebar.info("This app simulates a real-time feed to find the cheapest flights available right now. Click 'Refresh Data' to get the latest prices.")
st.sidebar.markdown("---")
st.sidebar.header("Example Questions")
st.sidebar.markdown("""
- What is the cheapest flight today?
- Find the cheapest ticket to Singapore.
- How much is the cheapest flight to London?
- Compare flight prices to Delhi and Mumbai.
""")


if 'flight_data' not in st.session_state or st.sidebar.button("🔄 Refresh Flight Data"):
    with st.spinner("Fetching latest flight data..."):
        st.session_state.flight_data = generate_realtime_flight_data()
        st.session_state.last_updated = datetime.now().strftime("%H:%M:%S")
    st.toast("Flight data has been updated!")

st.sidebar.markdown(f"**Data last updated:** {st.session_state.last_updated}")

try:
    
    retrieval_chain = get_rag_chain(st.session_state.flight_data)
   
    user_question = st.text_input("Ask a question to find the cheapest flight right now:")

    if user_question:
        with st.spinner("Searching real-time data..."):
            response = retrieval_chain.invoke({"input": user_question})
            st.success("Here is the information:")
            st.write(response["answer"])

            with st.expander("Show the real-time data used for this answer"):
                for i, doc in enumerate(response["context"]):
                    st.info(f"Source {i+1}:\n\n{doc.page_content}")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("Please ensure your GROQ_API_KEY is set correctly in a .env file.")

