import streamlit as st
import os
import google.generativeai as genai
import networkx as nx
from pyvis.network import Network
import tempfile
import json
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract relations using Gemini
def extract_relations_gemini(text):
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    Analyze the following text and extract all important entity-relation-entity triplets.

    ⚡ Return the output ONLY in this exact JSON array format, without any explanation, without code block, without comments.
    [
    {{"subject": "Entity1", "relation": "Relationship", "object": "Entity2"}},
    ...
    ]

    Text:
    {text}
    """
    response = model.generate_content(prompt)
    return response.text

def parse_relations(response_text):
    try:
        first_brace = response_text.find('[')
        if first_brace != -1:
            response_text = response_text[first_brace:]
        triples = json.loads(response_text)
        return triples
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        return []

def build_graph(triples):
    g = nx.DiGraph()
    for triplet in triples:
        g.add_node(triplet['subject'])
        g.add_node(triplet['object'])
        g.add_edge(triplet['subject'], triplet['object'], label=triplet['relation'])
    return g

def visualize_graph(g):
    net = Network(height="600px", width="100%", directed=True)
    net.from_nx(g)
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "graph.html")
    net.show_buttons(filter_=['physics'])
    net.save_graph(path)
    return path

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    return text

# Streamlit App
st.set_page_config(page_title="Knowledge Graph Generator", layout="wide")
st.title("🔗 Knowledge Representation Graph Generator")

input_type = st.radio("Select Input Type", ["Text", "PDF File"])

content = ""
if input_type == "Text":
    content = st.text_area("Paste your content here", height=300)
elif input_type == "PDF File":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        content = extract_text_from_pdf(uploaded_file)

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

                st.success("Knowledge Graph Generated Successfully!")
                st.components.v1.html(open(graph_path, 'r', encoding='utf-8').read(), height=600)
            else:
                st.warning("No relationships found.")
