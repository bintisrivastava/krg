import streamlit as st
import os
import google.generativeai as genai
import networkx as nx
from pyvis.network import Network
import tempfile
import json
import re
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------- Gemini Prompt for Advanced Extraction ----------------------
def extract_relations_gemini(text):
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    Extract detailed and context-aware entity–relation–entity triples from the following text.

    For each relation, include:
    - subject
    - subject_type (e.g., Person, Organization, Location)
    - relation
    - object
    - object_type
    - context (sentence or phrase it came from)

     Return output strictly in this JSON array format:
    [
      {{
        "subject": "Entity1",
        "subject_type": "Type",
        "relation": "Relation",
        "object": "Entity2",
        "object_type": "Type",
        "context": "Original sentence or short context here..."
      }},
      ...
    ]

    Text:
    {text}
    """
    response = model.generate_content(prompt)
    return response.text
    
def parse_relations(response_text):
    try:
        # Use regex to find the first JSON array (from first [ to matching ])
        pattern = re.compile(r'\[\s*(?:\{.*?\}\s*,?\s*)*\]')
        match = pattern.search(response_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in the response.")
        json_text = match.group()
        triples = json.loads(json_text)
        # Validate keys
        for t in triples:
            for key in ['subject', 'subject_type', 'relation', 'object', 'object_type', 'context']:
                if key not in t:
                    raise ValueError(f"Missing key: {key}")
        return triples
    except Exception as e:
        st.error(f"Failed to parse enriched JSON: {e}")
        return []

# ---------------------- Graph Builder ----------------------
def build_graph(triples):
    g = nx.DiGraph()
    for triplet in triples:
        subj = triplet['subject']
        obj = triplet['object']
        rel = triplet['relation']
        subj_type = triplet.get('subject_type', 'Entity')
        obj_type = triplet.get('object_type', 'Entity')
        context = triplet.get('context', '')

        g.add_node(subj, title=f"{subj_type}: {subj}", group=subj_type)
        g.add_node(obj, title=f"{obj_type}: {obj}", group=obj_type)
        g.add_edge(subj, obj, label=rel, title=context)
    return g

# ---------------------- Graph Visualization ----------------------
def visualize_graph(g):
    net = Network(height="600px", width="100%", directed=True)
    net.from_nx(g)
    net.set_options("""
    var options = {
      nodes: {
        shape: 'dot',
        size: 15,
        font: { size: 14 }
      },
      edges: {
        arrows: 'to',
        smooth: true,
        font: { align: 'top' }
      },
      physics: {
        enabled: true,
        solver: 'forceAtlas2Based'
      }
    }
    """)
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "graph.html")
    net.save_graph(path)
    return path

# ---------------------- Input Handlers ----------------------
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_url(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return "\n".join(p.get_text() for p in paragraphs)
    except Exception as e:
        st.error(f"Error fetching from URL: {e}")
        return ""

# ---------------------- Streamlit App ----------------------
st.set_page_config(page_title="Knowledge Graph Generator", layout="wide")
st.title("Knowledge Representation Graph")

input_type = st.radio("Select Input Type", ["Text", "PDF File", "URL"])
content = ""

if input_type == "Text":
    content = st.text_area("Paste your content here", height=300)

elif input_type == "PDF File":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        content = extract_text_from_pdf(uploaded_file)

elif input_type == "URL":
    url = st.text_input("Enter a URL")
    if url:
        content = extract_text_from_url(url)

if st.button("Generate Knowledge Graph"):
    if not content.strip():
        st.warning("Please provide content to analyze.")
    else:
        with st.spinner("Analyzing..."):
            response_text = extract_relations_gemini(content)
            triples = parse_relations(response_text)
            if triples:
                g = build_graph(triples)
                graph_path = visualize_graph(g)
                st.success("Knowledge Graph Generated!")
                st.components.v1.html(open(graph_path, 'r', encoding='utf-8').read(), height=600)
            else:
                st.warning("No relationships found")
