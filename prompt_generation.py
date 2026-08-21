"""
Prompt Studio Pro
------------------
Streamlit-App zum Verfeinern bestehender Prompts, Generieren mehrstufiger
Prompts aus Ideen und Verwalten einer Prompt-Template-Bibliothek.
Backend: LangChain + OpenRouter (ChatOpenRouter).
"""
import io
import json
import os
import re
from   typing import Tuple

import pandas as pd
import streamlit as st
from   PyPDF2 import PdfReader

from   langchain_openrouter import ChatOpenRouter
from   langchain_core.prompts import ChatPromptTemplate
from   langchain_core.output_parsers import StrOutputParser

##############################################################################
#  CONFIGURATION
##############################################################################

st.set_page_config(
    page_title="Prompt Generator & Refiner Studio",  page_icon="⚡", layout="wide", initial_sidebar_state="expanded",)
st.markdown(
    """
    <style>
    .stTextArea textarea { font-family: 'Fira Code', monospace; font-size: 14px; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: 600; }
    .highlight-card {
        padding: 16px;
        border-radius: 8px;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PROMPT_PATH ="./" 

##############################################################################
#  STATIC CONFIG / CATALOGS
##############################################################################
OPENROUTER_MODELS = [
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "inclusionai/ling-3.0-flash:free",
    "openrouter/free",
    # --- kostenpflichtige Modelle (Kosten pro Mio. Token als Kommentar) ---
    "mistralai/mistral-nemo",       # 2c
    "qwen/qwen3.7-flash",           # 3c
    "deepseek/deepseek-v4-flash",   # 8c
    "openai/gpt-5.6-luna",          # 10c
    "xiaomi/mimo-v2.5",             # 11c
    "deepseek/deepseek-v4-pro",     # 43c
    "openai/gpt-5.5",               # 500c
]

REFINER_PROFILES = [
    "General",
    "Detailed Expert",
    "Short & Concise",
    "Creative Writing",
    "Image Generation (Midjourney/DALL-E)",
    "Video Generation (Sora/Runway)",
    "Coding & Engineering",
    "Marketing & Growth",
    "Research & Analysis",
]

GENERATOR_DOMAINS = [
    "General Writing",
    "Image Generation",
    "Video Generation",
    "Coding & Architecture",
    "Marketing & Copywriting",
    "Academic Research",
]

PROMPT_TIERS = ("basic", "advanced", "expert")
TIER_LABELS = {
    "basic": "🟢 Basic Prompt",
    "advanced": "🟡 Advanced Prompt",
    "expert": "🔴 Expert Prompt",
}

##############################################################################
#  DATA / RESOURCE LOADING
##############################################################################
@st.cache_resource(show_spinner=False)
def get_openrouter_model(model_name: str, temperature: float) -> ChatOpenRouter:
    """Erstellt (und cached) das LangChain-Chat-Modell für OpenRouter."""
    api_key = st.secrets["OPEN_ROUTER_API_KEY"]
    if not api_key: raise ValueError("OPEN_ROUTER_API_KEY not found in environment/.env")
    return ChatOpenRouter(model=model_name, temperature=temperature, api_key=api_key)

@st.cache_data
def load_prompt_repository() -> pd.DataFrame:
    """Lädt die Prompt-Template-Bibliothek aus Excel, mit Fallback-Demodaten."""
    file_path = os.path.join(PROMPT_PATH, "Prompts_CBR.xlsx")

    if os.path.exists(file_path): df = pd.read_excel(file_path)
    else:
        df = pd.DataFrame(
            {
                "Name": ["Code Reviewer", "Copywriter Assistant", "Data Analyst"],
                "Prompt": [
                    "Review the following code for security and performance issues regarding {topic}:",
                    "Write an engaging blog post about {topic} targeting {topic2}.",
                    "Analyze the provided dataset with focus on {topic}.",
                ],
                "Kategorien": ["Coding, Tech", "Marketing, Creative", "Research, Analysis"],
            } )

    df["Kategorie_List"] = df["Kategorien"].astype(str).str.split(",").apply(lambda x: [k.strip() for k in x])
    return df

@st.cache_data(show_spinner=False)
def extract_pdf_text(file_bytes: bytes) -> Tuple[str, int]:
    """Extrahiert Text aus einer PDF-Datei (bytes, damit st.cache_data hashen kann)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted: text += extracted + "\n"
    return text, len(reader.pages)

##############################################################################
#  LLM HELPERS
##############################################################################
def run_llm(query: str, llm_model, spinner_text: str) -> str:
    """Führt einen Prompt gegen das LLM aus und gibt den reinen Text zurück."""
    with st.spinner(spinner_text):
        try:
            prompt_template = ChatPromptTemplate.from_template("{user_query}")
            chain = prompt_template | llm_model | StrOutputParser()
            return chain.invoke({"user_query": query})
        except Exception as exc:
            st.error(f"Fehler beim Ausführen des Prompts: {exc}")
            return ""

def extract_json_block(raw_text: str) -> str:
    """Entfernt optionale Markdown-Codefences (```json ... ```) robust und case-insensitiv."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned
    
##############################################################################
#  SESSION STATE DEFAULTS
##############################################################################
st.session_state.setdefault("model", OPENROUTER_MODELS[0])
st.session_state.setdefault("refined_prompt", "")
st.session_state.setdefault("generated_prompts", {"basic": "", "advanced": "", "expert": ""})
st.session_state.setdefault("test_response", "")
st.session_state.setdefault("test_response_tier", {"title": "", "text": ""})

##############################################################################
#  SIDEBAR
##############################################################################
st.sidebar.title("⚙️ Engine Settings")

selected_model = st.sidebar.selectbox("Model:", OPENROUTER_MODELS, index=0)
temperature = st.sidebar.slider("Creativity (Temperature)", 0.0, 1.0, 0.2, 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Document Context")
uploaded_file = st.sidebar.file_uploader("Upload PDF context", type=["pdf"])

pdf_text_context = ""
if uploaded_file is not None:
    pdf_text_context, page_count = extract_pdf_text(uploaded_file.getvalue())
    st.sidebar.success(f"Loaded {page_count} pages from PDF")

st.session_state.model = selected_model

try:
    llm_model = get_openrouter_model(selected_model, temperature)
except ValueError as exc:
    st.error(f"⚠️ {exc}. Bitte OPEN_ROUTER_API_KEY als Umgebungsvariable/.env-Eintrag setzen.")
    st.stop()

prompt_df = load_prompt_repository()

##############################################################################
#  MAIN INTERFACE
##############################################################################
st.title("⚡ Prompt Studio Pro")
st.caption("Modern AI Prompt Engineering, Refinement & Strategy Platform")

tab_refiner, tab_generator, tab_library = st.tabs(
    ["✨ Prompt Refiner", "🧠 Idea-to-Prompt Generator", "📚 Template Library"])

# ------------------------------------------
# TAB 1: PROMPT REFINER
# ------------------------------------------
with tab_refiner:
    st.markdown("### Refine & Optimize Existing Prompts")
    st.write("Transform raw, ambiguous prompts into highly structured, context-aware instructions.")

    col_input, col_config = st.columns([2, 1])

    with col_config:
        refiner_filter = st.selectbox("Optimization Profile / Persona:", REFINER_PROFILES)
        include_pdf = st.checkbox("Attach PDF Context to Prompt", value=bool(pdf_text_context), disabled=not bool(pdf_text_context),)

    with col_input:
        raw_prompt = st.text_area("Input Your Draft Prompt:", height=150,placeholder="e.g., Write a python script to scrape stock prices and email me...",)

    if st.button("🚀 Refine Prompt", type="primary", key="btn_refine"):
        if not raw_prompt.strip(): st.warning("Please enter a prompt to refine.")
        else:
            query = (
                f"You are a World-Class Prompt Engineer. Your task is to rewrite the user's raw prompt into a "
                f"professional, high-performing prompt optimized for LLMs. Target Profile: '{refiner_filter}'.\n"
                f"Structure your response with clear Role, Context, Instructions, Constraints, and Output Format "
                f"where applicable. Return ONLY the optimized prompt text without conversational preamble.\n\n"
                f"Raw Prompt: {raw_prompt}"
            )

            if include_pdf and pdf_text_context:
                excerpt = pdf_text_context[:3000]
                suffix = "..." if len(pdf_text_context) > 3000 else ""
                query += f"\n\nReference Context Document:\n{excerpt}{suffix}"

            refined_prompt = run_llm(query, llm_model, "Analyzing and optimizing prompt architecture...")
            if refined_prompt:
                st.session_state.refined_prompt = refined_prompt
                st.session_state.editable_prompt_area = refined_prompt
                st.session_state.test_response = ""

    if st.session_state.get("editable_prompt_area") or st.session_state.refined_prompt:
        st.markdown("---")
        st.subheader("Optimized Prompt Output")
        updated_prompt = st.text_area(label="Du kannst den Prompt hier direkt anpassen:", height=300, key="editable_prompt_area",)
        st.session_state.refined_prompt = updated_prompt

        col_act1, _ = st.columns([1, 4])
        with col_act1:
            if st.button("🧪 Test Prompt Now", use_container_width=True):
                if not st.session_state.refined_prompt.strip(): st.warning("Optimized prompt is empty - nothing to test.")
                else:
                    st.session_state.test_response = run_llm(
                        st.session_state.refined_prompt, llm_model, "Executing optimized prompt...")

        if st.session_state.test_response:
            st.markdown("---")
            st.markdown("#### 🧪 Execution Result:")
            st.info(st.session_state.test_response)

# ------------------------------------------
# TAB 2: PROMPT GENERATOR
# ------------------------------------------
with tab_generator:
    st.markdown("### Idea-to-Prompt Multi-Tier Generator")
    st.write("Convert raw concepts into multi-level prompts (Basic, Advanced, and Expert).")

    col_g1, col_g2 = st.columns([2, 1])

    with col_g2: gen_domain = st.selectbox("Target Domain:", GENERATOR_DOMAINS)

    with col_g1: concept_input = st.text_area(
        "Describe your idea or requirements:", height=130, 
        placeholder="e.g., An automated agent that summarizes daily financial news and generates a bulleted newsletter.",)

    if st.button("⚡ Generate Prompts (3 Tiers)", type="primary", key="btn_generate"):
        if not concept_input.strip(): st.warning("Please enter your target idea/requirements first.")
        else:
            gen_prompt = (
                "You are an expert prompt designer. Given a core idea and a target domain, generate 3 tiers of prompts:\n"
                "1. Basic Prompt: Direct, minimal instructions.\n"
                "2. Advanced Prompt: Includes role framing, task decomposition, and output formatting.\n"
                "3. Expert Prompt: Includes full system persona, step-by-step reasoning (Chain-of-Thought), "
                "edge-case handling, and strict formatting constraints.\n\n"
                "Respond STRICTLY in JSON format with keys: 'basic', 'advanced', 'expert'.\n\n"
                f"Domain: {gen_domain}\nRequirements/Idea: {concept_input}"
            )

            res = run_llm(gen_prompt, llm_model, "Engineering multi-tier prompt frameworks...")
            if res:
                try:
                    parsed_prompts = json.loads(extract_json_block(res))
                    st.session_state.generated_prompts = {tier: parsed_prompts.get(tier, "") for tier in PROMPT_TIERS }
                except (json.JSONDecodeError, AttributeError):
                    st.warning("Could not parse structured JSON response - showing raw output in 'Basic Prompt'.")
                    st.session_state.generated_prompts = {"basic": res, "advanced": "", "expert": ""}

    if st.session_state.generated_prompts["basic"]:
        st.markdown("---")
        tier_tabs = st.tabs([TIER_LABELS[t] for t in PROMPT_TIERS])

        for tier, tier_tab in zip(PROMPT_TIERS, tier_tabs):
            with tier_tab:
                area_key, prev_key = f"area_{tier}", f"prev_{tier}"

                if area_key not in st.session_state or st.session_state.generated_prompts.get(tier) != st.session_state.get(prev_key):
                    st.session_state[area_key] = st.session_state.generated_prompts.get(tier, "")
                    st.session_state[prev_key] = st.session_state[area_key]

                st.session_state.generated_prompts[tier] = st.text_area(
                    label=f"{tier.capitalize()} Prompt bearbeiten:", height=300, key=area_key)

                col_act, _ = st.columns([1, 4])
                with col_act:
                    if st.button(f"🧪 Test {tier.capitalize()} Prompt", key=f"btn_test_{tier}", use_container_width=True):
                        prompt_text = st.session_state.generated_prompts[tier]
                        if not prompt_text.strip(): st.warning(f"{tier.capitalize()} prompt is empty - nothing to test.")
                        else:
                            res = run_llm(prompt_text, llm_model, f"Executing {tier.capitalize()} Prompt...")
                            st.session_state.test_response_tier = {"title": f"{TIER_LABELS[tier]} Execution Result", "text": res, }

        if st.session_state.test_response_tier.get("text"):
            st.markdown("---")
            st.markdown(f"#### {st.session_state.test_response_tier['title']}")
            st.info(st.session_state.test_response_tier["text"])

# ------------------------------------------
# TAB 3: TEMPLATE LIBRARY
# ------------------------------------------
with tab_library:
    st.markdown("### Prompt Template Repository & Custom Execution")

    list_category = sorted(prompt_df["Kategorien"].astype(str).str.split(",").explode().str.strip().unique().tolist())
    categories = st.multiselect("Filter Category:", list_category)

    if categories: tmp_df = prompt_df[prompt_df["Kategorie_List"].apply(lambda kat: set(categories).issubset(set(kat)))].copy()
    else:          tmp_df = prompt_df.copy()
        
    tmp_df.insert(0, "Select", False)

    col_tbl, col_pdf_preview = st.columns([12, 4])

    with col_tbl:
        prompt_sel_df = st.data_editor(
            tmp_df[["Select", "Name", "Prompt", "Kategorien"]], hide_index=True,
            column_config={
                "Select": st.column_config.CheckboxColumn(required=True),
                "Prompt": st.column_config.TextColumn(width="large"),},
            key="template_editor",)

        selected_rows = prompt_sel_df[prompt_sel_df["Select"]]
        prompt_schablone = selected_rows.iloc[0]["Prompt"] if len(selected_rows) > 0 else ""

    with col_pdf_preview:
        st.subheader("PDF Text Stream")
        if pdf_text_context: st.text_area("Extracted Document Content", pdf_text_context, height=200)
        else:                st.info("Upload a PDF in the sidebar to stream context here.")

    st.markdown("#### Variable Placeholders")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        topic = st.text_input("Replace {topic}:", "")
        if topic: prompt_schablone = prompt_schablone.replace("{topic}", topic)
    with col_t2:
        topic2 = st.text_input("Replace {topic2}:", "")
        if topic2: prompt_schablone = prompt_schablone.replace("{topic2}", topic2)

    final_prompt = st.text_area("Execution Prompt:", height=120, value=prompt_schablone)

    if st.button("▶ Execute Prompt", type="primary"):
        if not final_prompt.strip(): st.warning("Please specify a prompt to execute.")
        else:
            exec_prompt = final_prompt
            if pdf_text_context: exec_prompt += f"\n\nContext:\n{pdf_text_context}"

            res = run_llm(exec_prompt, llm_model, "Processing request...")
            if res:
                st.markdown("### Response:")
                st.write(res)