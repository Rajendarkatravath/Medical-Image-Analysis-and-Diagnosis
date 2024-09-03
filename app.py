import streamlit as st
from PIL import Image
import io
import numpy as np
import faiss
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get the Groq API key from the environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Streamlit application interface
st.title("Medical Image Analysis and Diagnosis")
st.write("Upload an X-ray or MRI image to get a diagnosis, prescription, and precautions.")

# Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    # Initialize Llama model using ChatGroq from langchain_groq
    llm = ChatGroq(groq_api_key=api_key, model_name="llama3-70b-8192")  # Ensure the model name matches your configuration

    # Initialize Hugging Face embeddings via LangChain
    huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Define the FAISS index for embedding storage
    embedding_dimension = 384  # Example dimension for "all-MiniLM-L6-v2"; adjust if using a different model
    faiss_index = faiss.IndexFlatL2(embedding_dimension)

    # Define ChatPromptTemplate for Generating Descriptions, Prescriptions, and Precautions
    diagnosis_prompt_template = ChatPromptTemplate.from_template(
        template="You are a medical expert. Based on the following medical image description: {image_description}, provide a diagnosis."
    )
    
    # Define Chain using LangChain's LLMChain
    diagnosis_chain = LLMChain(llm=llm, prompt=diagnosis_prompt_template)

    # File uploader for medical images
    uploaded_file = st.file_uploader("Choose an X-ray or MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Load and preprocess the image
        image = Image.open(io.BytesIO(uploaded_file.read()))
        image = image.convert("RGB")  # Ensure the image is in RGB format

        # Display the uploaded image
        st.image(image, caption="Uploaded Medical Image", use_column_width=True)

        # Generate a caption for the image using a pre-trained model
        # Replace with an actual medical image captioning model for better accuracy
        image_text = "An X-ray image showing a lateral view of the human skull and brain structure."

        # Generate embeddings for the textual representation of the image
        image_embedding = huggingface_embeddings.embed_documents([image_text])[0]

        # Add the embedding to FAISS index
        faiss_index.add(np.array([image_embedding]))

        # Use the embedding to create a text prompt for the LLM
        diagnosis_input = {"image_description": image_text}

        # Predict diagnosis using the Llama model via Groq API
        prediction = diagnosis_chain.run(diagnosis_input)
        if not prediction:
            st.error("Model prediction failed.")
        else:
            # Display the results after the image
            st.subheader("Results")
            st.write(f"**Diagnosis:** {prediction}")

else:
    st.warning("Please enter your Groq API Key to proceed.")
