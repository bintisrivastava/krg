import streamlit as st
from pyvis.network import Network
import networkx as nx
import google.generativeai as genai
import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import tempfile
import json
import re
import os

# ---- Gemini Setup ----
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-2.0-flash")

# ---- Utility Functions ----
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_url(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.content, 'html.parser')
        return "\n".join(p.get_text() for p in soup.find_all(['p', 'h1', 'h2']))
    except:
        return ""

def extract_relations_with_gemini(text):
    prompt = (
        "Extract subject-predicate-object relations from the text below.\n"
        "Return as JSON: [{\"subject\": \"...\", \"relation\": \"...\", \"object\": \"...\"}]\n"
        f"Text:\n{text[:4000]}"
    )
    try:
        response = model.generate_content(prompt)
        json_text = re.search(r"\[.*\]", response.text, re.DOTALL).group(0)
        return json.loads(json_text)
    except:
        return []

def enrich_entity_with_wikidata(entity):
    try:
        query = f"""
        SELECT ?item ?itemLabel ?description WHERE {{
          ?item ?label "{entity}"@en.
          ?item schema:description ?description.
          FILTER(LANG(?description) = "en")
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT 1
        """
        r = requests.get("https://query.wikidata.org/sparql", params={"query": query}, headers={"Accept": "application/json"})
        results = r.json()["results"]["bindings"]
        if results:
            return results[0]["description"]["value"]
    except:
        return ""
    return ""

def build_knowledge_graph(text):
    relations = extract_relations_with_gemini(text)
    G = nx.MultiDiGraph()
    for triple in relations:
        subj, rel, obj = triple["subject"], triple["relation"], triple["object"]
        G.add_node(subj, label="Entity", description=enrich_entity_with_wikidata(subj))
        G.add_node(obj, label="Entity", description=enrich_entity_with_wikidata(obj))
        G.add_edge(subj, obj, label=rel)
    return G

def render_graph(G):
    net = Network(height="600px", width="100%", directed=True)
    for n, data in G.nodes(data=True):
        net.add_node(n, label=n, title=data.get("description", ""), group=data.get("label", "Entity"))
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, label=data["label"])
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp_file.name)
    return tmp_file.name

# ---- Streamlit UI ----
st.set_page_config(layout="wide")
st.title("🧠 Advanced Knowledge Graph Generator (Streamlit + Gemini)")

input_type = st.radio("Choose input type:", ["Text", "PDF", "URL"])

text_input = ""
if input_type == "Text":
    text_input = st.text_area("Enter text", height=200)
elif input_type == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        text_input = extract_text_from_pdf(uploaded_file)
elif input_type == "URL":
    url = st.text_input("Enter URL")
    if url:
        text_input = extract_text_from_url(url)

if st.button("Generate Knowledge Graph") and text_input.strip():
    with st.spinner("Extracting relations and building graph..."):
        G = build_knowledge_graph(text_input)
        graph_path = render_graph(G)
    st.success("Graph generated!")
    st.components.v1.html(open(graph_path, "r").read(), height=700, scrolling=True)
