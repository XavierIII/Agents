import langgraph
from langchain.agents import AgentType, initialize_agent
from langchain.llms import OpenAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
import streamlit as st
import spacy
import PyPDF2
import language_tool_python
from fpdf import FPDF

# Initialize AI Model
llm = OpenAI(model_name="gpt-4")

# Load NLP tools
nlp = spacy.load("en_core_web_sm")
spell_checker = language_tool_python.LanguageTool('en-US')

# Define Agents
class ContentAnalysisAgent:
    """
    This agent evaluates the relevance and impact of the resume content.
    """
    def analyze(self, resume_text):
        response = llm.predict(f"Analyze the resume for content relevance and provide feedback: {resume_text}")
        return response

class FormattingAgent:
    """
    This agent checks for ATS-friendly formatting and structure.
    """
    def check_formatting(self, resume_text):
        response = llm.predict(f"Evaluate the resume formatting for ATS compatibility: {resume_text}")
        return response

class KeywordOptimizationAgent:
    """
    This agent assesses the presence of industry-specific keywords.
    """
    def analyze_keywords(self, resume_text, job_description):
        response = llm.predict(f"Compare the resume with the job description and suggest keyword optimizations: \nResume: {resume_text}\nJob Description: {job_description}")
        return response

class ComplianceAgent:
    """
    This agent ensures the resume adheres to job description requirements.
    """
    def check_compliance(self, resume_text, job_description):
        response = llm.predict(f"Assess whether the resume matches the job description: \nResume: {resume_text}\nJob Description: {job_description}")
        return response

class GrammarCheckAgent:
    """
    This agent checks for grammatical and spelling errors in the resume.
    """
    def check_grammar(self, resume_text):
        return spell_checker.check(resume_text)

class PDFHandler:
    """
    Handles PDF resume uploads and extracts text.
    """
    @staticmethod
    def extract_text_from_pdf(uploaded_file):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return text

# LangGraph Workflow
class ATSResumeChecker:
    """
    This class orchestrates multiple AI agents using LangGraph to evaluate resumes.
    """
    def __init__(self):
        self.content_agent = ContentAnalysisAgent()
        self.formatting_agent = FormattingAgent()
        self.keyword_agent = KeywordOptimizationAgent()
        self.compliance_agent = ComplianceAgent()
        self.grammar_agent = GrammarCheckAgent()

    def evaluate_resume(self, resume_text, job_description):
        """Executes the ATS resume checking workflow."""
        results = {
            "Content Analysis": self.content_agent.analyze(resume_text),
            "Formatting Check": self.formatting_agent.check_formatting(resume_text),
            "Keyword Optimization": self.keyword_agent.analyze_keywords(resume_text, job_description),
            "Compliance Check": self.compliance_agent.check_compliance(resume_text, job_description),
            "Grammar & Spelling Check": self.grammar_agent.check_grammar(resume_text)
        }
        return results

# Streamlit UI
st.title("ATS Resume Checker with AI Agents & LangGraph")
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the job description here:")
resume_text = ""

if uploaded_file is not None:
    resume_text = PDFHandler.extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Resume Text:", value=resume_text, height=200)

if st.button("Analyze Resume"):
    ats_checker = ATSResumeChecker()
    results = ats_checker.evaluate_resume(resume_text, job_description)
    
    st.subheader("Analysis Results")
    for key, value in results.items():
        st.write(f"**{key}:**")
        st.write(value)
