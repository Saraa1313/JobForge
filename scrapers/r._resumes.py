import os
import pandas as pd
from pypdf import PdfReader
import google.generativeai as genai
import time

API_KEY = ""  # Removed before pushing the code

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FOLDER = os.path.join(BASE_DIR, '..', 'data', 'resumes')
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'data', 'real_resumes_extracted.csv')

genai.configure(api_key=API_KEY)

MODEL_NAME = 'gemini-2.5-flash' 

def get_model():
    return genai.GenerativeModel(MODEL_NAME)

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages[:2]: 
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def get_gemini_summary(text):
    model = get_model()
    
    prompt = f"""
    You are a resume parser. Summarize this resume text into EXACTLY one sentence using this specific format:
    "[Role] with [Years] years experience, holding a [Degree]. Expert in [Top 3-4 Skills]. [One key detailed achievement with metrics]."
    
    Rules:
    - Role: Choose closest fit (Software Engineer, Data Scientist, Product Manager, ML Engineer, Frontend Dev, Data Analyst).
    - Years: Estimate total years.
    - Achievement: Must be a specific work accomplishment.
    - No intro/outro. No Name.

    Resume Text:
    {text[:8000]}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        time.sleep(2)
        return None

def classify_role_from_summary(summary):
    s = summary.lower()
    if "product manager" in s: return "PM"
    if "data scientist" in s: return "DS"
    if "machine learning" in s: return "ML"
    if "data analyst" in s: return "DA"
    if "frontend" in s: return "FE"
    return "SWE"

def process_resumes():
    
    source_path = os.path.normpath(SOURCE_FOLDER)
    output_path = os.path.normpath(OUTPUT_FILE)

    print(f"📂 Looking for PDFs in: {source_path}")
    
    if not os.path.exists(source_path):
        print(f"Error: Folder not found!")
        return

    files = [f for f in os.listdir(source_path) if f.lower().endswith('.pdf')]
    if not files:
        print("No PDF files found in that folder.")
        return

    
    data = []
    
    for idx, filename in enumerate(files):
        print(f"   [{idx+1}/{len(files)}] Processing {filename}...")
        
        file_path = os.path.join(source_path, filename)
        raw_text = extract_text_from_pdf(file_path)
        
        if len(raw_text) < 50:
            print("Skipping Text too short/unreadable ")
            continue
            
        summary = get_gemini_summary(raw_text)
        
        if summary:
            role_code = classify_role_from_summary(summary)
            data.append({
                "resume_id": f"Real_R{idx+1}",
                "resume_role": role_code,
                "resume_text": summary
            })
            
            time.sleep(1) 
        else:
            print("Failed to generate summary.")

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"\n Saved to: {output_path}")
    else:
        print("\n Process finished but no data was saved.")

if __name__ == "__main__":
    process_resumes()