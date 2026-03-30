import streamlit as st
import PyPDF2
import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
import uuid
import pandas as pd
import io

st.set_page_config(page_title="AI Resume Screener | Mankar Official")
# ==========================================
# 1. API AND DATABASE SETUP
# ==========================================
load_dotenv()
# Direct Key 
genai.configure(api_key="enter_your_real_key")


# Keep vector database always accessible
chroma_client = chromadb.PersistentClient(path="./resume_db")
collection = chroma_client.get_or_create_collection(name="ai_resumes")


st.title("🤖 AI Resume Screener")
st.caption("Production RAG Pipeline | Built by Chitransh Mankar")
st.success("✅ Database connected and API ready!")


# Function to extract embeddings using Gemini
def get_embedding(text):
    response = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return response['embedding']


# Function for AI reasoning and evaluation summary
def get_ai_reason(jd, resume_text):
    # Use 'gemini-2.5-flash-lite' model for text generation
    model = genai.GenerativeModel('gemini-2.5-flash-lite') 
    
    # Advanced prompt engineering for structured output
    prompt = f"""
    You are an expert IT Recruiter. 
    Job Description: {jd}
    Candidate Resume: {resume_text}
    
    Task: Evaluate this candidate and provide output STRICTLY in this format:
    
    ⭐ **Match Score:** [Give a number between 1-10. Decimals allowed like 8.5/10. Below 3 if no skill match]
    
    📝 **AI Verdict:** [Explain in 2 lines why this score. Be brutally honest. Reject clearly if core skills missing.]
    """
    
    response = model.generate_content(prompt)
    return response.text


# Sidebar Admin Panel for database management
with st.sidebar:
    st.header("⚙️ Admin Controls")
    st.write("Clear previous data before starting new hiring cycle.")
    
    # Primary warning button for database reset
    if st.button("🧹 Clear Database ", type="primary"):
        with st.spinner("Clearing Database..."):
            try:
                # 1. Delete existing collection
                chroma_client.delete_collection("ai_resumes") 
                
                # 2. Create fresh empty collection
                collection = chroma_client.get_or_create_collection("ai_resumes")
                
                # 3. Success confirmation
                st.success("✅ Database cleared successfully! Ready for new hiring cycle. ✨")
                
            except Exception as e:
                st.error("Vector store already empty or error occurred.")


# ==========================================
# 2. PDF UPLOAD AND VECTOR STORE SAVE
# ==========================================
st.markdown("### 📄 Step 1: Upload Resume (PDF)")


# PDF uploader and candidate name input
uploaded_file = st.file_uploader("Upload candidate's PDF Resume", type="pdf")
candidate_name = st.text_input("Candidate Name:")


# Process uploaded PDF
if uploaded_file is not None:
    # Extract text from PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text() + "\n"
        
    st.info("✅ PDF text extracted successfully!")
    
    # Optional text preview (collapsible)
    with st.expander("👀 View Resume Text"):
        st.write(full_text)


    # Show save button only when name is provided
    if candidate_name:
        if st.button(f"Save {candidate_name} to Vector Store!"):
            with st.spinner("Generating embeddings and storing in vector database..."):
                try:
                    # 1. Generate embeddings using Gemini
                    embedding = get_embedding(full_text)
                    
                    # 2. Generate unique document ID
                    doc_id = str(uuid.uuid4())
                    
                    # 3. Store in ChromaDB vector store
                    collection.add(
                        embeddings=[embedding],     
                        documents=[full_text],     
                        metadatas=[{"name": candidate_name}], 
                        ids=[doc_id]
                    )
                    
                    st.success(f"🎉 Success! {candidate_name}'s resume stored in vector database!")
                except Exception as e:
                    st.error(f"⚠️ API Error: {e}")
    else:
        st.warning("⚠️ Please enter candidate name first to enable save button!")


# ==========================================
# 3. AI SEARCH ENGINE (VECTOR RETRIEVAL)
# ==========================================
st.markdown("---")
st.markdown("### 🔍 Step 2: AI Search (Find Best Candidates)")


# Job requirement input from HR
job_description = st.text_area("Enter Job Description (JD) or requirements (e.g., 'Need Python dev for automation'):")


if st.button("🔍 Find Top Candidates!"):
    if job_description:
        with st.spinner("AI searching vector store for best matches..."):
            try:
                # 1. Generate embeddings for job description
                jd_embedding = get_embedding(job_description)
                
                # 2. Query vector store (return top 3 matches)
                results = collection.query(
                    query_embeddings=[jd_embedding],
                    n_results=3 
                )
                
                # 3. Display results if vector store has data
                if results['documents'][0]:
                    st.success("✅ Here are the top matching candidates!")
                    
                    # Initialize report data list
                    report_data = []
                    
                    # Process each matching result
                    for i in range(len(results['documents'][0])):
                        candidate_name = results['metadatas'][0][i]['name']
                        resume_text = results['documents'][0][i]
                        
                        with st.expander(f"👤 Candidate: {candidate_name}"):
                            with st.spinner("AI Recruiter evaluating candidate..."):
                                ai_evaluation = get_ai_reason(job_description, resume_text)
                                st.markdown("### 🤖 AI Evaluation")
                                st.success(ai_evaluation) 
                            
                            st.divider() 
                            st.caption("**📄 Original Resume Snippet:**")
                            st.caption(resume_text[:300] + "...") 
                            
                            # Clean AI evaluation for Excel export
                            clean_ai_eval = ai_evaluation.replace('\n', ' ').replace('**', '')
                            clean_ai_eval = clean_ai_eval.replace('⭐', 'Score:').replace('📝', 'Verdict:')
                            
                            # Clean resume snippet for export (remove PDF formatting artifacts)
                            clean_snippet = resume_text[:200].replace('\n', ' ').replace('Text to save as PDF:', ':').replace('Name:', '')

                            report_data.append({
                                "Candidate Name": candidate_name,
                                "AI Evaluation": clean_ai_eval,
                                "Resume Snippet": clean_snippet.strip() + "..."
                            })


                    # Excel Download Section
                    if report_data:
                        st.write("---")
                        df = pd.DataFrame(report_data)
                        
                        # Create Excel file in memory buffer
                        buffer = io.BytesIO()
                        
                        # Write DataFrame to Excel using openpyxl engine
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='AI Shortlist')
                        
                        # Streamlit download button for Excel
                        st.download_button(
                            label="📊 Download HR Report (Excel)",
                            data=buffer.getvalue(),
                            file_name='HR_Shortlist_Report.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        ) 
                        
                        
                else:
                    st.warning("No resumes in vector store or no matches found!")
                    
            except Exception as e:
                st.error(f"⚠️ Search error: {e}")
    else:
        st.warning("⚠️ Please enter job requirements first!")
    
# FOOTER
st.markdown("---")
st.markdown(
    """
    **👨‍💻 Developed By**  
      Chitransh Mankar - Aspiring Software Engineer & AI Enthusiast | 
    From Bhopal  
    """,
    unsafe_allow_html=True
)
