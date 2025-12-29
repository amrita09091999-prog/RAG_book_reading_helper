import streamlit as st 
import requests

upload_pdf_url = "http://127.0.0.1:8000/novel_upload"
ask_questions_url = "http://127.0.0.1:8000/ask_questions"
evaluation_url = "http://127.0.0.1:8000/evaluate"

st.title("Welcome! This is your AI Novel Reading Assistant!")
st.write("Ask any question related to your current novel, and we will do our best to answer your question")
st.markdown("<small>Do know that AI can make mistakes</small>", unsafe_allow_html=True)

st.subheader("Upload your Novel - only PDF files are accepted, ignore if its already uploaded once")

uploaded_file = st.file_uploader(
    "Upload your novel (PDF)",
    type=["pdf"]
)

if uploaded_file:
    st.write(f"File name: {uploaded_file.name}")
    files = {
    "file": (uploaded_file.name, uploaded_file, "application/pdf")
    }
    with st.spinner("Uploading your novel..."):
        response = requests.post(url =upload_pdf_url, files=files)

    if response.status_code == 200:
        st.success("Novel uploaded successfully!")
        st.caption(f"Size: {uploaded_file.size / 1024:.2f} KB")
    else:
        st.error("Upload failed")

st.subheader("Ask questions to the Assistant")
query = st.text_input(
    "Enter your question",
    placeholder="e.g. How is Saidar different from Saidin?"
)
book_name = st.text_input(
    "Enter the book name you want the answers from",
    placeholder="e.g. Wheel of time book 3"
)
ask_button = st.button("Ask")
if ask_button:
    if not query.strip():
        st.warning("Please enter the question first")
    if not book_name.strip():
        st.warning("Please enter the name of the book")
    else:
        payload = {
            'query': query,
            'book_name':book_name
        }
        st.spinner("Thinking...")
        response = requests.post(url = ask_questions_url, json = payload)

    if response.status_code==200:
        st.markdown("##Answer")
        st.write(response.json())
    else:
        st.error("Something went wrong while fetching the answer.")

st.subheader("Response Evaluation")
evaluate_button = st.button("Evaluate")
if evaluate_button:
    st.spinner("Thinking...")
    response=  requests.post(url = evaluation_url)
    if response.status_code==200:
        st.write(response.json())

