import streamlit as st
import os
import google.generativeai as genai
import networkx as nx
from pyvis.network import Network
import tempfile
import json
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract entity-relation-entity triplets
def extract_relations_gemini(text):
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    Extract all meaningful subject-relation-object triples from the following content.

    Return only a valid JSON array like:
    [
        {{"subject": "Entity1", "relation": "Relationship", "object": "Entity2"}},
        ...
    ]

    Content:
    {text}
    """
    response = model.generate_content(prompt)
    return response.text

# Parse the Gemini response into usable JSON
def parse_relations(response_text):
    try:
        response_text = response_text[response_text.find('['):]
        return json.loads(response_text)
    except Exception as e:
        st.error(f"Failed to parse output: {e}")
        return []

# Create a graph from triplets
def build_graph(triples):
    graph = nx.DiGraph()
    for item in triples:
        graph.add_node(item['subject'])
        graph.add_node(item['object'])
        graph.add_edge(item['subject'], item['object'], label=item['relation'])
    return graph

# Create styled PyVis HTML from NetworkX graph
def visualize_graph(graph):
    net = Network(height="600px", width="100%", directed=True, bgcolor="#111", font_color="white")
    net.from_nx(graph)

    for node in net.nodes:
        node['color'] = "#00ccff"
        node['shape'] = "dot"
        node['size'] = 20

    for edge in net.edges:
        edge['color'] = "#ffdd00"
        edge['arrows'] = "to"
        edge['font'] = {'size': 12, 'color': 'white'}

    net.set_options("""
    var options = {
      "nodes": {
        "borderWidth": 2,
        "shadow": true
      },
      "edges": {
        "smooth": false,
        "shadow": true
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -20000,
          "springLength": 150
        },
        "minVelocity": 0.75
      }
    }
    """)

    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "styled_graph.html")
    net.save_graph(path)
    return path

# Extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# Extract text from URL
def extract_text_from_url(url):
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all('p')
    return "\n".join(p.get_text() for p in paragraphs)

# Streamlit UI setup
st.set_page_config(page_title="Knowledge Graph Generator", layout="wide")
st.markdown("<h1 style='color:#00ccff'>Knowledge Graph Generator</h1>", unsafe_allow_html=True)
st.markdown("Binti Srivastava")
st.markdown("PEE203 Project")

# Input method selection
input_type = st.radio("Choose Input Type", ["Text Input", "📎 Upload PDF", "🌐 URL Input"])

# Input handling
content = ""
if input_type == "Text Input":
    content = st.text_area("Paste your content here:", height=300)
elif input_type == " Upload PDF":
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file:
        content = extract_text_from_pdf(uploaded_file)
elif input_type == " URL Input":
    url = st.text_input("Enter the URL:")
    if url:
        try:
            content = extract_text_from_url(url)
        except Exception as e:
            st.error(f"Failed to retrieve or parse URL: {e}")

# Main button
if st.button("Generate Knowledge Graph"):
    if not content.strip():
        st.warning("Please provide text, upload a PDF, or enter a URL.")
    else:
        with st.spinner("Extracting relationships..."):
            response_text = extract_relations_gemini(content)
            triples = parse_relations(response_text)

            if triples:
                g = build_graph(triples)
                graph_path = visualize_graph(g)
                st.success("Knowledge Graph Generated Successfully!")
                st.components.v1.html(open(graph_path, 'r', encoding='utf-8').read(), height=650)
            else:
                st.warning("No meaningful relationships were found.")
