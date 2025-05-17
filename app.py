import streamlit as st
import os
import google.generativeai as genai
import networkx as nx
from pyvis.network import Network
import tempfile
import json
import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Wikidata Ontology Enrichment ---
def enrich_entity_with_wikidata(entity):
    query = f'''
    SELECT ?item ?itemLabel ?description WHERE {{
      ?item ?label "{entity}"@en.
      ?item schema:description ?description.
      FILTER (lang(?description) = "en")
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }} LIMIT 1
    '''
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/json"}
    response = requests.get(url, params={'query': query}, headers=headers)

    try:
        data = response.json()
        bindings = data['results']['bindings']
        if bindings:
            return bindings[0]['description']['value']
    except Exception:
        pass
    return "No description found."

# --- Gemini Relation Extraction ---
def extract_relations_gemini(text):
    model = genai.GenerativeModel('gemini-1.5-pro')  # use the most powerful model
    prompt = f"""
    Extract key entity–relation–entity triples from the text below.

    Format:
    [
      {{"subject": "Entity1", "relation": "Relation", "object": "Entity2"}},
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

# --- Graph Construction with Enrichment ---
def build_graph(triples):
    g = nx.DiGraph()
    node_data = {}

    for triplet in triples:
        subj = triplet['subject']
        obj = triplet['object']
        rel = triplet['relation']

        # Add nodes with ontology enrichment
        if subj not in node_data:
            node_data[subj] = enrich_entity_with_wikidata(subj)
        if obj not in node_data:
            node_data[obj] = enrich_entity_with_wikidata(obj)

        g.add_node(subj, title=node_data[subj])
        g.add_node(obj, title=node_data[obj])
        g.add_edge(subj, obj, label=rel)

    return g

# --- Interactive Graph Visualization ---
def visualize_graph(g):
    net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.from_nx(g)

    for node in net.nodes:
        node['title'] = g.nodes[node['id']].get('title', "")
        node['label'] = node['id']
        node['shape'] = 'dot'
        node['size'] = 20

    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "graph.html")
    net.show_buttons(filter_=['physics'])
    net.save_graph(path)
    return path

# --- PDF Text Extraction ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    return text

# --- Streamlit UI ---
st.set_page_config(page_title="Advanced Knowledge Graph Generator", layout="wide")
st.title("🔍 Advanced Knowledge Graph Generator")

input_type = st.radio("Select Input Type", ["Text", "PDF File", "Web URL"])

content = ""
if input_type == "Text":
    content = st.text_area("Paste your content here", height=300)
elif input_type == "PDF File":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        content = extract_text_from_pdf(uploaded_file)
elif input_type == "Web URL":
    url = st.text_input("Enter the URL:")
    if url and st.button("Fetch Content"):
        try:
            response = requests.get(url)
            content = response.text
        except Exception as e:
            st.error(f"Failed to fetch content: {e}")

if st.button("Generate Knowledge Graph"):
    if not content.strip():
        st.warning("Please provide content to analyze.")
    else:
        with st.spinner("Analyzing and building graph..."):
            response_text = extract_relations_gemini(content)
            triples = parse_relations(response_text)

            if triples:
                g = build_graph(triples)
                graph_path = visualize_graph(g)

                st.success("✅ Knowledge Graph Generated!")
                st.components.v1.html(open(graph_path, 'r', encoding='utf-8').read(), height=600)
            else:
                st.warning("⚠️ No relationships found.")
