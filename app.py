import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Step 1: Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Step 2: Named Entity Recognition (NER) with spaCy
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "SKILL", "TITLE", "WORK_OF_ART"]]
    return entities

# Step 3: Load GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Step 4: Set padding token
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token  # Use eos_token as pad_token

# Step 5: Use GPT-2 to generate embeddings by using hidden states from the model
def embed_text(text):
    # Tokenize the text
    inputs = gpt2_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Get the model's output
    with torch.no_grad():
        outputs = gpt2_model(**inputs, output_hidden_states=True)
    
    # Extract the hidden states (the second-to-last layer, as it's a good representation)
    hidden_states = outputs.hidden_states[-2]  # Get second to last layer (for better representation)
    
    # We average the token embeddings to get a single vector for the whole input text
    embeddings = hidden_states.mean(dim=1).squeeze().numpy()
    
    return embeddings

# Step 6: Calculate cosine similarity between resume and job description
def calculate_similarity(resume_text, job_description_text):
    resume_embedding = embed_text(resume_text)
    job_description_embedding = embed_text(job_description_text)
    
    similarity_score = cosine_similarity([resume_embedding], [job_description_embedding])[0][0]
    return similarity_score

# Step 7: Generate a summary based on matching entities and similarity score
def generate_summary_with_gpt2(similarity_score, matching_entities):
    # Convert similarity score to percentage
    similarity_percentage = similarity_score * 100
    
    # Create a focused prompt based on desired output
    prompt = (
        f"The similarity score between the candidate's resume and the job description is {similarity_percentage:.2f}%, indicating a strong match. "
        f"Key matching skills include expertise in {', '.join(matching_entities)}. "
    )
    
    # Generate summary with GPT-2
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(
        inputs,
        max_new_tokens=100,  # Limit output length to keep the summary concise
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.2,
        top_p=0.9
    )
    summary = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return summary

# Streamlit UI code
st.title("Resume and Job Description Matching")

# Sidebar for file upload
st.sidebar.header("Upload Files")
resume_file = st.sidebar.file_uploader("Upload Resume PDF", type="pdf")
job_desc_file = st.sidebar.file_uploader("Upload Job Description PDF", type="pdf")

if resume_file and job_desc_file:
    # Extract text from uploaded PDFs
    resume_text = extract_text_from_pdf(resume_file)
    job_description_text = extract_text_from_pdf(job_desc_file)

    # Extract entities from resume and job description
    resume_entities = extract_entities(resume_text)
    job_description_entities = extract_entities(job_description_text)

    # Find matching entities
    matching_entities = list(set(resume_entities).intersection(job_description_entities))

    # Calculate cosine similarity
    similarity_score = calculate_similarity(resume_text, job_description_text)

    # Generate and display the summary
    output_summary = generate_summary_with_gpt2(similarity_score, matching_entities)
    st.subheader("Summary")
    st.write(output_summary)

else:
    st.warning("Please upload both a resume and a job description PDF to proceed.")
