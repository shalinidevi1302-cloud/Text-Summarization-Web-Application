import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("🤖 AI Text Summarizer")
st.write("Summarize long articles and paragraphs using Hugging Face Transformers.")

@st.cache_resource
def load_summarizer():
    model_name = "sshleifer/distilbart-cnn-12-6"
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "summarization",
        model=model_name,
        device=device
    )

with st.sidebar:
    st.header("About the Project")
    st.write("This application uses a pre-trained Hugging Face model.")
    st.write("Task: Text Summarization")
    st.write("Model: DistilBART CNN")
    st.write("Framework: Transformers Pipeline")

    st.subheader("Summary Settings")
    min_length = st.slider("Minimum summary length", 10, 60, 20)
    max_length = st.slider("Maximum summary length", 30, 150, 60)

st.subheader("Enter Your Text")

input_text = st.text_area(
    "Paste your article or paragraph here:",
    height=250,
    placeholder="Enter a long article or paragraph to summarize..."
)

if input_text.strip():
    word_count = len(input_text.split())
    st.caption(f"Input word count: {word_count}")

if st.button("✨ Generate Summary", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text before generating a summary.")
    elif len(input_text.split()) < 20:
        st.warning("Please enter a longer text with at least 20 words.")
    elif min_length >= max_length:
        st.error("Maximum summary length must be greater than minimum length.")
    else:
        try:
            with st.spinner("Loading AI model and generating summary..."):
                summarizer = load_summarizer()

                result = summarizer(
                    input_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )

                summary = result[0]["summary_text"]

            st.success("Summary generated successfully!")

            st.subheader("📄 Generated Summary")
            st.write(summary)

            original_words = len(input_text.split())
            summary_words = len(summary.split())

            col1, col2, col3 = st.columns(3)

            col1.metric("Original Words", original_words)
            col2.metric("Summary Words", summary_words)

            if original_words > 0:
                reduction = (1 - summary_words / original_words) * 100
                col3.metric("Text Reduction", f"{reduction:.1f}%")

            st.download_button(
                label="⬇️ Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )

        except Exception as error:
            st.error("Unable to generate the summary.")
            st.write("Please check your internet connection and try again.")
            st.caption(f"Error details: {error}")

st.divider()
st.caption("Built using Python, Streamlit, and Hugging Face Transformers.")