from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import os
import time

#  Load environment variables
load_dotenv()

# Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")

#  API Key validation
if not gemini_api_key:
    st.error("❌ API key not found. Please check your .env file!")
    st.stop()

#  Streamlit app configuration
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖", layout="wide")
st.title("🚀 LangChain + Gemini ")

# Sidebar settings
st.sidebar.header("⚙️ Settings")

#  Temperature slider
temperature = st.sidebar.slider(
    "Temperature (creativity level)", 0.0, 1.0, 0.7, 0.05
)

# Token Usage Tracking Toggle
track_tokens = st.sidebar.checkbox("Track Token Usage", value=True)

#  User input
input_text = st.text_area("💬 Ask your question:")

#  LangChain Gemini LLM (default model: gemini-2.0-flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  
    google_api_key=gemini_api_key,
    temperature=temperature
)

#  Clear button
if st.button("🧹 Clear"):
    input_text = ""
    st.rerun()

#  Generate response
if st.button("⚡ Get Answer"):
    if input_text.strip():
        with st.spinner("⏳ Generating response..."):
            start_time = time.time()  # 🌟 Track response time

            try:
                # 🌟 Invoke the LLM
                response = llm.invoke(input_text)

                # Handle different response formats
                if hasattr(response, 'content'):
                    output = response.content  # For BaseMessage object
                elif isinstance(response, dict):
                    output = response.get("text", "No text content found.")  # For dict format
                else:
                    output = str(response)

                # Display the response
                st.subheader("✅ Response:")
                st.write(output)

                # 🌟 Display token usage and response time
                end_time = time.time()
                st.success(f"🕒 Response time: {round(end_time - start_time, 2)} seconds")

                # ✅ Token usage tracking
                if track_tokens:
                    token_count = len(output.split())  # Rough token count
                    st.info(f"🔢 Token count: {token_count}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    else:
        st.warning("⚠️ Please enter a question!")
