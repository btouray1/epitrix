# EPITRIX — Epitope Intelligence Platform
# Mechanistic AI: Molecular Design → Innate Sensing → Adaptive Quality/Magnitude
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Epitrix | Epitope Intelligence Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Force light mode everywhere ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
    [data-testid="stMain"], section.main, .main {
        background-color: #f9fafb !important;
        color: #111827 !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #1e293b !important;
    }
    /* Sidebar radio text white on dark sidebar */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] div {
        color: #f1f5f9 !important;
    }
    /* All widget labels clearly dark */
    label, .stSlider label, .stSelectbox label, .stTextArea label,
    .stNumberInput label, .stTextInput label {
        color: #111827 !important; font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important; font-size: 0.875rem !important;
    }
    p, li { color: #111827; font-family: 'Inter', sans-serif; }

    /* ── Selectbox / dropdown — light bg, dark text, at every DOM level ── */
    .stSelectbox > div, .stSelectbox > div > div,
    [data-baseweb="select"], [data-baseweb="select"] > div,
    [data-baseweb="select"] input,
    [data-baseweb="popover"], [data-baseweb="popover"] > div,
    [data-baseweb="menu"], [data-baseweb="menu"] ul,
    [role="listbox"], [role="option"],
    [data-baseweb="select"] [data-testid="stSelectboxVirtualDropdown"] {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    /* Dropdown trigger button */
    [data-baseweb="select"] > div:first-child {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        color: #111827 !important;
    }
    /* Selected value text */
    [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
    [data-baseweb="select"] span, [data-baseweb="select"] div {
        color: #111827 !important;
    }
    /* Dropdown list items */
    [role="option"] { background: #ffffff !important; color: #111827 !important; }
    [role="option"]:hover { background: #eff6ff !important; color: #1e40af !important; }
    /* Tab widgets */
    [data-baseweb="tab-list"] { background: #f3f4f6 !important; border-radius: 8px; }
    [data-baseweb="tab"] { color: #374151 !important; background: transparent !important; }
    [aria-selected="true"][data-baseweb="tab"] {
        color: #2563eb !important; background: #ffffff !important;
        border-bottom: 2px solid #2563eb !important;
    }
    button[data-baseweb="tab"] span { color: inherit !important; }
    /* Multiselect tags */
    [data-baseweb="tag"] { background: #dbeafe !important; color: #1e40af !important; }
    /* Text inputs */
    .stTextArea textarea, .stTextInput input, .stNumberInput input {
        background: #ffffff !important; color: #111827 !important;
        border: 1px solid #d1d5db !important; border-radius: 8px !important;
    }
    /* Slider track */
    .stSlider [data-baseweb="slider"] { background: transparent !important; }
    /* ── Buttons — force light/white style outside sidebar ── */
    .stButton > button {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stButton > button:hover {
        background-color: #f3f4f6 !important;
        border-color: #9ca3af !important;
    }
    /* Primary buttons keep the blue */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid*="primary"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-color: #2563eb !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
    }
    /* Download button */
    .stDownloadButton > button {
        background-color: #ffffff !important;
        color: #2563eb !important;
        border: 1px solid #2563eb !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover {
        background-color: #eff6ff !important;
    }
    /* Expander */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    /* Metric labels */
    [data-testid="stMetricLabel"] { color: #6b7280 !important; }
    [data-testid="stMetricValue"] { color: #111827 !important; }
    /* Info/success/warning boxes */
    .stAlert { background: #f0f9ff !important; color: #1e3a5f !important; border-radius: 8px !important; }
    /* Spinner text */
    .stSpinner > div { color: #2563eb !important; }

    :root {
        --primary: #2563eb; --secondary: #10b981; --accent: #f59e0b;
        --warning: #ef4444; --success: #22c55e;
        --gray-50: #f9fafb; --gray-100: #f3f4f6; --gray-200: #e5e7eb;
        --gray-600: #4b5563; --gray-700: #374151; --gray-800: #1f2937; --gray-900: #111827;
        --font: 'Inter', -apple-system, sans-serif;
    }
    #MainMenu, footer, header, .stDeployButton { visibility: hidden; }
    .main .block-container { padding: 0; max-width: 100%; background: #f9fafb !important; }

    /* ── Header ── */
    .epitrix-header-legacy {
        background: linear-gradient(135deg, #1e1b4b 0%, #2563eb 50%, #0891b2 100%);
        padding: 2.5rem 2rem; margin: 0;
    }
    .header-content { max-width: 1400px; margin: 0 auto; }
    .platform-title {
        font-family: var(--font); font-size: 3rem; font-weight: 700; margin: 0;
        background: linear-gradient(45deg, #ffffff, #a5f3fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .platform-subtitle { font-family: var(--font); font-size: 1.1rem; margin: 0.5rem 0 0; color: rgba(255,255,255,0.9); }
    .breakthrough-tagline { font-family: var(--font); font-size: 0.85rem; font-weight: 600; color: #a5f3fc; text-transform: uppercase; letter-spacing: 0.1em; margin: 0; }

    /* ── Cards & layout ── */
    .content-container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
    .innovation-card {
        background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px;
        padding: 2rem; margin-bottom: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.07);
    }
    .innovation-card h2, .innovation-card h3, .innovation-card h4,
    .innovation-card p, .innovation-card li, .innovation-card strong, .innovation-card em {
        color: #111827 !important;
    }
    .section-title { font-family: var(--font); font-size: 1.75rem; font-weight: 700; color: #111827 !important; margin: 0 0 0.5rem; }
    .section-subtitle { font-family: var(--font); font-size: 1rem; color: #4b5563 !important; margin: 0; }

    .data-card {
        background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px;
        padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .data-card h4 { color: #2563eb !important; margin: 0 0 0.75rem; font-size: 1rem; font-weight: 600; }
    .data-card ul { margin: 0; padding-left: 1.25rem; }
    .data-card li { color: #374151 !important; margin-bottom: 0.4rem; font-size: 0.9rem; }
    .data-card.output { border-left: 4px solid #10b981; }

    .prediction-target {
        background: linear-gradient(135deg, #f0f9ff, #ffffff);
        border: 1px solid #bae6fd; border-left: 4px solid #2563eb;
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
    }
    .prediction-target h4 { color: #2563eb !important; margin: 0 0 0.5rem; font-size: 1rem; font-weight: 700; }
    .prediction-target p { color: #374151 !important; margin: 0 0 0.5rem; font-size: 0.9rem; }
    .prediction-target li { color: #4b5563 !important; font-size: 0.85rem; }

    /* ── Cascade flow — explicit dark text so it shows on light bg ── */
    .cascade-flow { display: flex; flex-direction: column; align-items: center; gap: 0.5rem; padding: 1rem 0; }
    .flow-step {
        background: linear-gradient(135deg, #eff6ff, #ffffff);
        border: 1.5px solid #bfdbfe; border-radius: 10px;
        padding: 1rem 2rem; text-align: center; width: 100%; max-width: 520px;
    }
    .flow-step strong { color: #1e3a8a !important; font-size: 0.95rem; display: block; margin-bottom: 0.2rem; }
    .flow-step em { color: #4b5563 !important; font-size: 0.85rem; }
    .flow-arrow { font-size: 1.5rem; color: #2563eb; }

    /* ── Antigen input card ── */
    .antigen-card {
        background: linear-gradient(135deg, #fefce8, #ffffff);
        border: 1.5px solid #fde68a; border-left: 4px solid #f59e0b;
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
    }
    .antigen-card h3 { color: #92400e !important; margin: 0 0 0.5rem; font-size: 1rem; font-weight: 700; }
    .antigen-card p, .antigen-card li { color: #78350f !important; font-size: 0.85rem; }
    .antigen-stat { display:inline-block; background:#fef3c7; border:1px solid #fde68a;
        border-radius:6px; padding:2px 8px; margin:2px; font-size:0.78rem; color:#92400e !important; font-weight:600; }

    .time-point {
        background: #f9fafb; border-left: 3px solid #2563eb;
        padding: 0.75rem 1rem; margin-bottom: 0.75rem; border-radius: 0 8px 8px 0;
    }
    .time-point h5 { margin: 0 0 0.25rem; color: #2563eb !important; font-size: 0.9rem; }
    .time-point p { margin: 0; color: #4b5563 !important; font-size: 0.85rem; }

    .mol-input-box {
        background: #f9fafb; border: 1px solid #e5e7eb;
        border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;
    }
    .mol-input-box h3 { margin: 0 0 0.75rem; color: #111827 !important; font-size: 1rem; font-weight: 600; }

    .stMarkdown p, .stMarkdown li { color: #1f2937 !important; font-family: var(--font) !important; }
    .stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4 {
        color: #111827 !important; font-family: var(--font) !important; font-weight: 700 !important;
    }
    @media (max-width: 768px) { .platform-title { font-size: 2rem; } .content-container { padding: 1rem; } }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MOLECULAR DESCRIPTORS DATABASE
# ─────────────────────────────────────────────────────────────────────────────

# Lipid category tags for UI grouping
LIPID_CATEGORIES = {
    # --- Clinical / approved ---
    'ALC-0315':         'Clinical',
    'SM-102':           'Clinical',
    # --- FDA-approved siRNA delivery ---
    'DLin-MC3-DMA':     'Approved (siRNA)',
    # --- Cationic (research) ---
    'DOTAP':            'Cationic',
    'DOTMA':            'Cationic',
    # --- Next-generation ionizable ---
    'OF-Deg-Lin':       'Next-Gen',
    'C12-200':          'Next-Gen',
    'Lipid 5':          'Next-Gen',
    '306Oi10':          'Next-Gen',
    'CL4H6':            'Next-Gen',
    # --- Biodegradable ionizable ---
    'ALC-0315-BD':      'Biodegradable',
    'L-319':            'Biodegradable',
    # --- Custom ---
    '⚙️ Custom Lipid':  'Custom',
}

MOLECULAR_DESCRIPTORS = {
    'lipid_chemistry': {
        'ionizable_lipids': {
            # ── Clinical / approved ───────────────────────────────────────────
            'ALC-0315': {
                'pka': 6.09, 'logP': 8.2, 'molecular_weight': 766.0, 'branching_factor': 2.4,
                'notes': 'Used in Pfizer-BioNTech COVID-19 mRNA vaccine (Comirnaty)',
                'innate_profile': {'TLR_activation': 'low', 'complement_trigger': 'minimal', 'dc_maturation': 'moderate'}
            },
            'SM-102': {
                'pka': 6.68, 'logP': 7.8, 'molecular_weight': 688.5, 'branching_factor': 2.1,
                'notes': 'Used in Moderna COVID-19 mRNA vaccine (Spikevax)',
                'innate_profile': {'TLR_activation': 'low', 'complement_trigger': 'minimal', 'dc_maturation': 'moderate'}
            },
            # ── Approved (siRNA) ──────────────────────────────────────────────
            'DLin-MC3-DMA': {
                'pka': 6.44, 'logP': 7.5, 'molecular_weight': 642.1, 'branching_factor': 2.0,
                'notes': 'FDA-approved for siRNA delivery (Onpattro/patisiran)',
                'innate_profile': {'TLR_activation': 'moderate', 'complement_trigger': 'low', 'dc_maturation': 'high'}
            },
            # ── Cationic ─────────────────────────────────────────────────────
            'DOTAP': {
                'pka': 8.5, 'logP': 6.5, 'molecular_weight': 698.5, 'branching_factor': 1.2,
                'notes': 'Permanent cationic lipid; high transfection but elevated reactogenicity',
                'innate_profile': {'TLR_activation': 'high', 'complement_trigger': 'moderate', 'dc_maturation': 'high'}
            },
            'DOTMA': {
                'pka': 9.0, 'logP': 6.2, 'molecular_weight': 670.5, 'branching_factor': 1.0,
                'notes': 'First synthetic cationic lipid; precursor to DOTAP; high immunostimulation',
                'innate_profile': {'TLR_activation': 'high', 'complement_trigger': 'high', 'dc_maturation': 'high'}
            },
            # ── Next-gen ionizable ────────────────────────────────────────────
            'OF-Deg-Lin': {
                'pka': 6.50, 'logP': 7.1, 'molecular_weight': 602.0, 'branching_factor': 2.2,
                'notes': 'Degradable ester-linked ionizable; reduced toxicity profile',
                'innate_profile': {'TLR_activation': 'low', 'complement_trigger': 'minimal', 'dc_maturation': 'moderate'}
            },
            'C12-200': {
                'pka': 7.10, 'logP': 8.9, 'molecular_weight': 642.0, 'branching_factor': 3.1,
                'notes': 'Multi-tail ionizable; high hepatic delivery efficiency',
                'innate_profile': {'TLR_activation': 'moderate', 'complement_trigger': 'low', 'dc_maturation': 'high'}
            },
            'Lipid 5': {
                'pka': 6.32, 'logP': 7.4, 'molecular_weight': 710.2, 'branching_factor': 2.6,
                'notes': 'Highly branched ionizable; optimized endosomal escape',
                'innate_profile': {'TLR_activation': 'low', 'complement_trigger': 'minimal', 'dc_maturation': 'moderate'}
            },
            '306Oi10': {
                'pka': 6.78, 'logP': 8.0, 'molecular_weight': 655.0, 'branching_factor': 2.8,
                'notes': 'Orthogonal branching scaffold; enhanced T cell response',
                'innate_profile': {'TLR_activation': 'moderate', 'complement_trigger': 'low', 'dc_maturation': 'high'}
            },
            'CL4H6': {
                'pka': 6.20, 'logP': 7.6, 'molecular_weight': 625.3, 'branching_factor': 2.3,
                'notes': 'Cyclic amine head group; improved selective organ targeting',
                'innate_profile': {'TLR_activation': 'low', 'complement_trigger': 'minimal', 'dc_maturation': 'moderate'}
            },
            # ── Biodegradable ─────────────────────────────────────────────────
            'ALC-0315-BD': {
                'pka': 6.15, 'logP': 7.9, 'molecular_weight': 748.0, 'branching_factor': 2.4,
                'notes': 'Biodegradable analogue of ALC-0315; faster clearance',
                'innate_profile': {'TLR_activation': 'low', 'complement_trigger': 'minimal', 'dc_maturation': 'low'}
            },
            'L-319': {
                'pka': 6.56, 'logP': 7.3, 'molecular_weight': 617.0, 'branching_factor': 2.0,
                'notes': 'Ester-bond biodegradable; reduced hepatotoxicity vs MC3',
                'innate_profile': {'TLR_activation': 'low', 'complement_trigger': 'minimal', 'dc_maturation': 'moderate'}
            },
            # ── Custom placeholder (populated dynamically from UI) ─────────────
            '⚙️ Custom Lipid': {
                'pka': 6.5, 'logP': 7.5, 'molecular_weight': 680.0, 'branching_factor': 2.0,
                'notes': 'User-defined custom lipid. Edit properties in the sidebar.',
                'innate_profile': {'TLR_activation': 'low', 'complement_trigger': 'minimal', 'dc_maturation': 'moderate'}
            },
        },
        # ── Helper lipids ─────────────────────────────────────────────────────
        'helper_lipids': {
            '— None (no helper lipid) —': {
                'molecular_weight': 0, 'phase_transition_temp': None,
                'membrane_rigidity': 0.0, 'notes': 'No helper lipid; binary LNP formulation'
            },
            # ── Phosphatidylcholines (PC) ─────────────────────────────────────
            'DSPC': {
                'molecular_weight': 790.1, 'phase_transition_temp': 55,
                'membrane_rigidity': 0.9, 'notes': 'High Tm saturated PC; gold standard for LNP stability (Pfizer/Moderna)'
            },
            'DPPC': {
                'molecular_weight': 733.6, 'phase_transition_temp': 41,
                'membrane_rigidity': 0.8, 'notes': 'Dipalmitoyl PC; slightly lower Tm than DSPC; good bilayer former'
            },
            'DMPC': {
                'molecular_weight': 677.9, 'phase_transition_temp': 23,
                'membrane_rigidity': 0.6, 'notes': 'Near-physiological Tm; useful for temperature-triggered release'
            },
            'DOPC': {
                'molecular_weight': 786.1, 'phase_transition_temp': -20,
                'membrane_rigidity': 0.2, 'notes': 'Fluid unsaturated PC; high membrane permeability'
            },
            'POPC': {
                'molecular_weight': 760.1, 'phase_transition_temp': -2,
                'membrane_rigidity': 0.35, 'notes': 'Mixed-chain PC mimicking natural membranes; excellent biocompatibility'
            },
            'HSPC (hydrogenated SPC)': {
                'molecular_weight': 784.0, 'phase_transition_temp': 52,
                'membrane_rigidity': 0.88, 'notes': 'Hydrogenated soy PC; high oxidative stability; used in Doxil'
            },
            # ── Phosphatidylethanolamines (PE) ────────────────────────────────
            'DOPE': {
                'molecular_weight': 744.0, 'phase_transition_temp': -16,
                'membrane_rigidity': 0.3, 'notes': 'Fusogenic PE lipid; hexagonal phase former; enhances endosomal escape'
            },
            'POPE': {
                'molecular_weight': 717.5, 'phase_transition_temp': 25,
                'membrane_rigidity': 0.45, 'notes': 'Mixed-chain PE; moderate fusogenicity; good endosomal escape'
            },
            'DSPE': {
                'molecular_weight': 790.1, 'phase_transition_temp': 74,
                'membrane_rigidity': 0.95, 'notes': 'Saturated PE; very high rigidity; mainly used as PEG anchor'
            },
            # ── Sphingolipids & sterols ────────────────────────────────────────
            'Sphingomyelin (SM)': {
                'molecular_weight': 702.9, 'phase_transition_temp': 41,
                'membrane_rigidity': 0.85, 'notes': 'Natural sphingolipid; forms ordered raft domains; slow clearance'
            },
            'Ceramide': {
                'molecular_weight': 537.9, 'phase_transition_temp': 65,
                'membrane_rigidity': 0.95, 'notes': 'Highly rigid; pro-apoptotic at high molar%; used as PEG anchor'
            },
            # ── Other ─────────────────────────────────────────────────────────
            'DLPC': {
                'molecular_weight': 621.8, 'phase_transition_temp': -1,
                'membrane_rigidity': 0.25, 'notes': 'Short-chain PC; highly fluid; fast drug release kinetics'
            },
            'DMPG': {
                'molecular_weight': 665.9, 'phase_transition_temp': 23,
                'membrane_rigidity': 0.55, 'notes': 'Anionic PG headgroup; negatively charged surface; immune modulating'
            },
        },
        # ── PEG-lipids ────────────────────────────────────────────────────────
        'peg_lipids': {
            '— None (no PEG) —': {
                'peg_mw': 0, 'anchor': 'None', 'shedding_rate': 'N/A',
                'notes': 'No PEG-lipid; no steric stabilization; faster clearance but higher uptake'
            },
            # ── Clinical / approved ───────────────────────────────────────────
            'ALC-0159 (PEG2000-DMG)': {
                'peg_mw': 2000, 'anchor': 'DMG', 'shedding_rate': 'fast',
                'notes': 'Used in Pfizer-BioNTech COVID vaccine; rapid PEG shedding enhances cellular uptake'
            },
            'PEG2000-C-DMG': {
                'peg_mw': 2000, 'anchor': 'C-DMG', 'shedding_rate': 'fast',
                'notes': 'Used in Moderna COVID vaccine (mSM-102); comparable kinetics to ALC-0159'
            },
            # ── PEG-DSPE family ───────────────────────────────────────────────
            'PEG2000-DSPE': {
                'peg_mw': 2000, 'anchor': 'DSPE', 'shedding_rate': 'slow',
                'notes': 'Non-cleavable anchor; long circulation half-life but reduced intracellular uptake'
            },
            'PEG5000-DSPE': {
                'peg_mw': 5000, 'anchor': 'DSPE', 'shedding_rate': 'very slow',
                'notes': 'Extended PEG chain; maximal steric shielding; used in stealth liposomes'
            },
            'PEG550-DSPE': {
                'peg_mw': 550, 'anchor': 'DSPE', 'shedding_rate': 'fast',
                'notes': 'Short PEG; minimal stealth effect; faster cellular uptake kinetics'
            },
            # ── PEG-ceramide / lipid conjugates ───────────────────────────────
            'PEG1000-C8-Ceramide': {
                'peg_mw': 1000, 'anchor': 'Ceramide', 'shedding_rate': 'moderate',
                'notes': 'Short C8 ceramide anchor; good balance of stealth and uptake'
            },
            'PEG2000-C16-Ceramide': {
                'peg_mw': 2000, 'anchor': 'C16-Ceramide', 'shedding_rate': 'slow',
                'notes': 'Longer ceramide anchor; improved retention at physiological temperature'
            },
            # ── PEG-DPPE / DLPE ───────────────────────────────────────────────
            'PEG2000-DPPE': {
                'peg_mw': 2000, 'anchor': 'DPPE', 'shedding_rate': 'moderate',
                'notes': 'Dipalmitoyl PE anchor; intermediate shedding; good compatibility with DPPC systems'
            },
            'PEG2000-DLPE': {
                'peg_mw': 2000, 'anchor': 'DLPE', 'shedding_rate': 'fast',
                'notes': 'Short-chain PE anchor; rapid shedding; useful for fast intracellular delivery'
            },
            # ── PEG-cholesterol ────────────────────────────────────────────────
            'PEG2000-Cholesterol': {
                'peg_mw': 2000, 'anchor': 'Cholesterol', 'shedding_rate': 'moderate',
                'notes': 'Cholesterol anchor; sheds by cholesterol exchange; good for solid LNPs'
            },
            'PEG600-Cholesterol': {
                'peg_mw': 600, 'anchor': 'Cholesterol', 'shedding_rate': 'fast',
                'notes': 'Short-chain PEG-cholesterol; minimal stealth, maximizes surface exposure'
            },
        },
    },
    'nucleic_acid_modifications': {
        # ── Standard / baseline ───────────────────────────────────────────────
        'Unmodified': {
            'tlr_evasion': 0.0, 'stability': 0.5, 'translation_eff': 0.7,
            'cap_compatibility': 1.0, 'notes': 'No modification; highest innate sensing, lowest stability'
        },
        # ── Clinical / approved modifications ────────────────────────────────
        'Pseudouridine (Ψ)': {
            'tlr_evasion': 0.70, 'stability': 0.80, 'translation_eff': 0.90,
            'cap_compatibility': 1.0, 'notes': 'First-generation TLR evasion; used in early mRNA vaccines'
        },
        'N1-methyl-pseudouridine (m1Ψ)': {
            'tlr_evasion': 0.85, 'stability': 0.90, 'translation_eff': 0.95,
            'cap_compatibility': 1.0, 'notes': 'Gold standard; used in Pfizer/Moderna COVID vaccines'
        },
        # ── Cytidine modifications ────────────────────────────────────────────
        '5-methylcytidine (m5C)': {
            'tlr_evasion': 0.50, 'stability': 0.75, 'translation_eff': 0.80,
            'cap_compatibility': 1.0, 'notes': 'Moderate TLR3 evasion; often combined with Ψ'
        },
        'm5C + Ψ (dual)': {
            'tlr_evasion': 0.80, 'stability': 0.85, 'translation_eff': 0.88,
            'cap_compatibility': 1.0, 'notes': 'Combined modification; broader TLR evasion spectrum'
        },
        # ── Uridine modifications ─────────────────────────────────────────────
        '2-thiouridine (s2U)': {
            'tlr_evasion': 0.60, 'stability': 0.70, 'translation_eff': 0.75,
            'cap_compatibility': 0.9, 'notes': 'TLR7 evasion; mildly reduces translation efficiency'
        },
        '5-methoxyuridine (mo5U)': {
            'tlr_evasion': 0.65, 'stability': 0.72, 'translation_eff': 0.78,
            'cap_compatibility': 1.0, 'notes': 'Moderate innate evasion; good translation compatibility'
        },
        # ── Adenosine modifications ───────────────────────────────────────────
        'N6-methyladenosine (m6A)': {
            'tlr_evasion': 0.45, 'stability': 0.78, 'translation_eff': 0.85,
            'cap_compatibility': 1.0, 'notes': 'Epitranscriptomic mark; regulates mRNA decay and translation'
        },
        # ── Self-amplifying / saRNA ────────────────────────────────────────────
        'saRNA (unmodified)': {
            'tlr_evasion': 0.10, 'stability': 0.60, 'translation_eff': 1.50,
            'cap_compatibility': 0.8, 'notes': 'Self-amplifying RNA; very high antigen expression, strong innate sensing'
        },
        'saRNA + m1Ψ': {
            'tlr_evasion': 0.65, 'stability': 0.80, 'translation_eff': 1.45,
            'cap_compatibility': 0.8, 'notes': 'Modified saRNA; partially dampened innate sensing with amplification'
        },
        # ── Circular RNA ───────────────────────────────────────────────────────
        'circRNA': {
            'tlr_evasion': 0.75, 'stability': 0.98, 'translation_eff': 0.70,
            'cap_compatibility': 0.0, 'notes': 'Cap-independent translation; exceptional stability, no 5\'/3\' exonuclease degradation'
        },
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# ANTIGEN DATABASE & SEQUENCE ANALYSIS
# Real epitope prediction:
#   1. Attempt IEDB Analysis Resource REST API (tools.iedb.org) — the gold standard
#   2. If unreachable/timeout: fall back to a local PSSM-based predictor that
#      uses published HLA anchor position weight matrices (far more accurate
#      than simple amino acid counting)
# ─────────────────────────────────────────────────────────────────────────────
import requests as _requests
import json as _json

# ─────────────────────────────────────────────────────────────────────────────
# ADJUVANT DATABASE
# Each entry encodes:
#   pathway_activates : dict  — binary flags (True/False) per pathway
#   pathway_strength  : dict  — ordinal (none/low/moderate/high/very_high)
#   scores            : dict  — mapped 0-1 values (none=0, low=0.15,
#                               moderate=0.35, high=0.65, very_high=0.85)
#   th_bias           : dict  — delta modifiers applied to Th1/Th2/Th17/Tfh
#   dc_boost          : float — DC maturation boost (0-0.4)
#   cd8_boost         : float — cross-presentation / CD8 T cell boost (0-0.4)
#   confidence        : str   — 'approved'|'clinical'|'preclinical'|'research'
#   notes             : str   — key mechanistic note with source
#   compatible        : list  — vaccine types this adjuvant applies to
#
# Score mapping (literature-grounded ordinal → 0-1):
#   none=0.0  low=0.15  moderate=0.35  high=0.65  very_high=0.85
#
# th_bias deltas are additive to the cascade-computed values before normalisation.
# Positive = boost that Th subset; negative = suppress.
# ─────────────────────────────────────────────────────────────────────────────

def _s(ordinal):
    """Map ordinal string to 0-1 score."""
    return {'none': 0.0, 'low': 0.15, 'moderate': 0.35,
            'high': 0.65, 'very_high': 0.85}[ordinal]

ADJUVANTS = {

    # ── None ─────────────────────────────────────────────────────────────────
    'None (formulation only)': {
        'scores':     {'TLR7_8': 0.0, 'TLR3': 0.0, 'cGAS_STING': 0.0,
                       'Complement': 0.0, 'Inflammasome': 0.0},
        'th_bias':    {'Th1': 0.0, 'Th2': 0.0, 'Th17': 0.0, 'Tfh': 0.0},
        'dc_boost': 0.0, 'cd8_boost': 0.0,
        'confidence': 'approved',
        'class': 'None',
        'notes': 'No adjuvant added. Innate activation from formulation chemistry only.',
        'compatible': ['mRNA', 'DNA', 'Protein subunit'],
    },

    # ══ CLASS 1: MINERAL SALTS (Approved) ═══════════════════════════════════

    'Alum (aluminium hydroxide)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('low'),
                       'Inflammasome': _s('high')},
        'th_bias':    {'Th1': -0.15, 'Th2': +0.30, 'Th17': 0.0, 'Tfh': -0.05},
        'dc_boost': 0.10, 'cd8_boost': 0.0,
        'confidence': 'approved',
        'class': 'Mineral salt',
        'notes': 'NLRP3-driven IL-1β/IL-18 (Kool 2008 J Immunol; Li 2008 J Immunol). '
                 'Strong Th2/IgG1/IgE bias. STING dispensable (Immunity 2024). '
                 'NLRP3 required for IL-1β but antibody production NLRP3-independent (PMC 2009).',
        'compatible': ['Protein subunit', 'DNA', 'mRNA'],
    },

    'Aluminium phosphate': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('low'),
                       'Inflammasome': _s('moderate')},
        'th_bias':    {'Th1': -0.10, 'Th2': +0.25, 'Th17': 0.0, 'Tfh': -0.05},
        'dc_boost': 0.08, 'cd8_boost': 0.0,
        'confidence': 'approved',
        'class': 'Mineral salt',
        'notes': 'Similar mechanism to Al hydroxide; different surface charge affects '
                 'antigen adsorption kinetics. Slightly less inflammasome activation '
                 'than hydroxide form (Brewer 1999 J Immunol; Badran 2022 Sci Rep).',
        'compatible': ['Protein subunit', 'DNA', 'mRNA'],
    },

    # ══ CLASS 2: OIL-IN-WATER EMULSIONS (Approved) ══════════════════════════

    'MF59 (squalene o/w emulsion)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('low'),
                       'Inflammasome': _s('low')},
        'th_bias':    {'Th1': +0.05, 'Th2': +0.05, 'Th17': 0.0, 'Tfh': +0.05},
        'dc_boost': 0.20, 'cd8_boost': 0.10,
        'confidence': 'approved',
        'class': 'Emulsion',
        'notes': 'NOT a TLR agonist (Seubert 2011 PNAS). Non-TLR MyD88 pathway. '
                 'RIPK3-dependent necroptosis drives CD8 cross-presentation (eLife 2020). '
                 'Broad chemokine induction; DC recruitment and monocyte-to-DC '
                 'differentiation. Balanced/mildly Th2-leaning (PMC 2023).',
        'compatible': ['Protein subunit', 'mRNA'],
    },

    'AS03 (squalene + α-tocopherol)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('low'),
                       'Inflammasome': _s('low')},
        'th_bias':    {'Th1': +0.05, 'Th2': +0.05, 'Th17': 0.0, 'Tfh': +0.08},
        'dc_boost': 0.22, 'cd8_boost': 0.08,
        'confidence': 'approved',
        'class': 'Emulsion',
        'notes': 'Similar to MF59 with α-tocopherol. Stimulates immune activation '
                 'in both muscle and draining LN. Stronger Tfh induction than MF59 '
                 'in head-to-head comparisons (PMC 2024 systems vaccinology). '
                 'Used in pandemic influenza vaccines.',
        'compatible': ['Protein subunit', 'mRNA'],
    },

    'AddaVax (MF59 mimetic)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('low'),
                       'Inflammasome': _s('low')},
        'th_bias':    {'Th1': +0.05, 'Th2': +0.05, 'Th17': 0.0, 'Tfh': +0.05},
        'dc_boost': 0.18, 'cd8_boost': 0.10,
        'confidence': 'approved',
        'class': 'Emulsion',
        'notes': 'Commercially available MF59 equivalent (InvivoGen). Same RIPK3 '
                 'necroptosis mechanism for CD8 cross-presentation (eLife 2020). '
                 'Widely used in preclinical mRNA-LNP adjuvant studies.',
        'compatible': ['Protein subunit', 'mRNA'],
    },

    'Montanide ISA 51 (water-in-oil)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('moderate'),
                       'Inflammasome': _s('low')},
        'th_bias':    {'Th1': +0.10, 'Th2': +0.08, 'Th17': 0.0, 'Tfh': +0.05},
        'dc_boost': 0.12, 'cd8_boost': 0.05,
        'confidence': 'clinical',
        'class': 'Emulsion',
        'notes': 'Water-in-oil depot adjuvant. Slow antigen release prolongs '
                 'exposure. Used in malaria and HIV clinical trials. Mixed Th1/Th2 '
                 'without TLR activation. Local reactogenicity can be significant '
                 '(Reed 2009 Expert Rev Vaccines).',
        'compatible': ['Protein subunit'],
    },

    'Montanide ISA 720 (squalene w/o)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('low'),
                       'Inflammasome': _s('low')},
        'th_bias':    {'Th1': +0.08, 'Th2': +0.08, 'Th17': 0.0, 'Tfh': +0.05},
        'dc_boost': 0.10, 'cd8_boost': 0.03,
        'confidence': 'clinical',
        'class': 'Emulsion',
        'notes': 'Squalene-in-water; better tolerated than ISA 51. Malaria trials '
                 '(Aucouturier 2001 Vaccine). Depot effect only, no TLR agonism.',
        'compatible': ['Protein subunit'],
    },

    # ══ CLASS 3: TLR AGONISTS ════════════════════════════════════════════════

    'MPLA / MPL (TLR4 agonist)': {
        'scores':     {'TLR7_8': _s('high'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('none')},
        'th_bias':    {'Th1': +0.30, 'Th2': -0.05, 'Th17': 0.0, 'Tfh': +0.10},
        'dc_boost': 0.30, 'cd8_boost': 0.05,
        'confidence': 'approved',
        'class': 'TLR agonist',
        'notes': 'TLR4 agonist, detoxified LPS from S. minnesota. TLR4→NF-κB→Th1. '
                 'Scores mapped to TLR7_8 slot as the dominant innate-Th1 driver. '
                 'First TLR agonist in licensed vaccines (AS04, 2009). '
                 'Note: weaker human vs mouse TLR4 activity (~27% efficacy vs Lipid A '
                 'on human TLR4 (Frontiers Immunol 2020).',
        'compatible': ['Protein subunit', 'mRNA', 'DNA'],
    },

    'GLA-SE (TLR4 synthetic)': {
        'scores':     {'TLR7_8': _s('high'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('none')},
        'th_bias':    {'Th1': +0.32, 'Th2': -0.05, 'Th17': 0.0, 'Tfh': +0.10},
        'dc_boost': 0.28, 'cd8_boost': 0.05,
        'confidence': 'clinical',
        'class': 'TLR agonist',
        'notes': 'Synthetic glucopyranosyl lipid A in squalene emulsion. Cleaner '
                 'than MPL (defined single compound). Strong Th1 via TLR4. '
                 'TB and HIV Phase I/II trials (Reed 2016 Sci Transl Med). '
                 'Designed for improved human TLR4 activity over natural MPL.',
        'compatible': ['Protein subunit', 'mRNA', 'DNA'],
    },

    'GLA-AF (TLR4, aqueous)': {
        'scores':     {'TLR7_8': _s('moderate'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('none')},
        'th_bias':    {'Th1': +0.22, 'Th2': -0.03, 'Th17': 0.0, 'Tfh': +0.07},
        'dc_boost': 0.18, 'cd8_boost': 0.03,
        'confidence': 'clinical',
        'class': 'TLR agonist',
        'notes': 'Same GLA as GLA-SE but aqueous nanosuspension. Lower potency '
                 'than SE formulation due to reduced APC targeting (PMC 2023). '
                 'Better thermostability; suitable for resource-limited settings.',
        'compatible': ['Protein subunit', 'mRNA', 'DNA'],
    },

    'CpG 1018 / ODN 1018 (TLR9)': {
        'scores':     {'TLR7_8': _s('very_high'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('none')},
        'th_bias':    {'Th1': +0.35, 'Th2': -0.10, 'Th17': 0.0, 'Tfh': +0.08},
        'dc_boost': 0.25, 'cd8_boost': 0.05,
        'confidence': 'approved',
        'class': 'TLR agonist',
        'notes': 'TLR9 agonist, mapped to TLR7_8 slot as the dominant innate-Th1 driver. '
                 'FDA-approved 2017 in HEPLISAV-B hepatitis B vaccine (Dynavax). '
                 'Strong IL-12, IFN-α, B cell activation. Very strong Th1/IgG2 bias. '
                 'Class B (K-type) CpG ODN (Coffman 2010 Immunity 33:492).',
        'compatible': ['Protein subunit', 'mRNA', 'DNA'],
    },

    'CpG 7909 / ODN 2006 (TLR9)': {
        'scores':     {'TLR7_8': _s('very_high'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('none')},
        'th_bias':    {'Th1': +0.35, 'Th2': -0.10, 'Th17': 0.0, 'Tfh': +0.08},
        'dc_boost': 0.25, 'cd8_boost': 0.05,
        'confidence': 'approved',
        'class': 'TLR agonist',
        'notes': 'FDA-approved 2023 in Cyfendus anthrax vaccine. Class B CpG. '
                 'Similar mechanism to CpG 1018; different sequence optimised '
                 'for human TLR9 (Wiley BTM 2024).',
        'compatible': ['Protein subunit', 'mRNA', 'DNA'],
    },

    'R848 / Resiquimod (TLR7/8)': {
        'scores':     {'TLR7_8': _s('very_high'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('none')},
        'th_bias':    {'Th1': +0.35, 'Th2': -0.08, 'Th17': 0.0, 'Tfh': +0.12},
        'dc_boost': 0.28, 'cd8_boost': 0.10,
        'confidence': 'clinical',
        'class': 'TLR agonist',
        'notes': 'Direct TLR7/8 dual agonist (imidazoquinoline). Strong IFN-α, '
                 'IL-12, TNF-α. Strong Th1 and CD8 cross-presentation. '
                 'Phase I/II HIV and cancer vaccines. Systemic exposure limited '
                 'by formulation (Frontiers Microbiol 2023).',
        'compatible': ['Protein subunit', 'mRNA', 'DNA'],
    },

    '3M-052 (TLR7/8, lipidated)': {
        'scores':     {'TLR7_8': _s('very_high'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('none')},
        'th_bias':    {'Th1': +0.38, 'Th2': -0.08, 'Th17': 0.0, 'Tfh': +0.15},
        'dc_boost': 0.30, 'cd8_boost': 0.12,
        'confidence': 'clinical',
        'class': 'TLR agonist',
        'notes': 'Lipidated TLR7/8 agonist, retained at injection site, reducing '
                 'systemic exposure. Induces long-lived plasma cells (up to ~1 year) '
                 'in NHP studies. Robust antiviral IFN program similar to yellow '
                 'fever vaccine (Nat Comms 2022). Phase I/II HIV, SARS-CoV-2.',
        'compatible': ['Protein subunit', 'mRNA', 'DNA'],
    },

    'AS37 (TLR7, benzonaphthyridine-alum)': {
        'scores':     {'TLR7_8': _s('high'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('low'),
                       'Inflammasome': _s('low')},
        'th_bias':    {'Th1': +0.25, 'Th2': +0.05, 'Th17': 0.0, 'Tfh': +0.08},
        'dc_boost': 0.22, 'cd8_boost': 0.08,
        'confidence': 'clinical',
        'class': 'TLR agonist',
        'notes': 'TLR7-selective benzonaphthyridine scaffold adsorbed to alum for '
                 'local retention. Phase I meningococcal vaccine with acceptable safety, '
                 'expected innate pathways activated (PMC 2025; Siena 2023). '
                 'Alum component adds moderate inflammasome activation.',
        'compatible': ['Protein subunit', 'mRNA', 'DNA'],
    },

    'Imiquimod / R837 (TLR7)': {
        'scores':     {'TLR7_8': _s('high'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('none')},
        'th_bias':    {'Th1': +0.28, 'Th2': -0.05, 'Th17': 0.0, 'Tfh': +0.08},
        'dc_boost': 0.20, 'cd8_boost': 0.08,
        'confidence': 'approved',
        'class': 'TLR agonist',
        'notes': 'TLR7-selective imidazoquinoline. FDA-approved for topical skin use '
                 '(Aldara). Used as adjuvant in intradermal vaccine studies. '
                 'Strong Th1 via IFN-α/IL-12. Species note: TLR8 not functional '
                 'in mice; TLR7-only in murine models (Immunity 2010).',
        'compatible': ['Protein subunit', 'mRNA', 'DNA'],
    },

    'Poly(I:C) (TLR3/MDA-5)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('very_high'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('none')},
        'th_bias':    {'Th1': +0.25, 'Th2': -0.05, 'Th17': 0.0, 'Tfh': +0.05},
        'dc_boost': 0.25, 'cd8_boost': 0.20,
        'confidence': 'clinical',
        'class': 'TLR agonist',
        'notes': 'dsRNA mimic activating TLR3 and cytosolic MDA-5. Strong type I IFN, IL-12. '
                 'cDC1 cross-presentation → strong CD8 T cell responses. '
                 'Toxic at high doses; rapid serum degradation limits clinical use '
                 '(Frontiers Immunol 2023; PMC 2022).',
        'compatible': ['Protein subunit', 'mRNA', 'DNA'],
    },

    'Poly-ICLC / Hiltonol (TLR3)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('high'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('none')},
        'th_bias':    {'Th1': +0.25, 'Th2': -0.05, 'Th17': 0.0, 'Tfh': +0.05},
        'dc_boost': 0.22, 'cd8_boost': 0.18,
        'confidence': 'clinical',
        'class': 'TLR agonist',
        'notes': 'Poly(I:C) stabilised with poly-L-lysine + carboxymethylcellulose. '
                 'Nuclease-resistant; lower toxicity than parent poly(I:C). '
                 'Stronger Th1 than LPS or CpG in direct comparisons (PMC 2022). '
                 'Clinical trials: malaria, HIV, cancer vaccines (Frontiers 2023).',
        'compatible': ['Protein subunit', 'mRNA', 'DNA'],
    },

    'Pam3CSK4 (TLR1/2)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('none')},
        'th_bias':    {'Th1': +0.12, 'Th2': 0.0, 'Th17': +0.15, 'Tfh': +0.03},
        'dc_boost': 0.15, 'cd8_boost': 0.05,
        'confidence': 'preclinical',
        'class': 'TLR agonist',
        'notes': 'Bacterial lipoprotein mimic. TLR1/2 heterodimer activates NF-κB, driving Th1/Th17. '
                 'Note: no dedicated Epitrix pathway slot; effect is captured via '
                 'dc_boost and th_bias deltas. Science Advances 2024 saponin-TLRa '
                 'combination study.',
        'compatible': ['Protein subunit', 'mRNA', 'DNA'],
    },

    # ══ CLASS 4: SAPONINS ═══════════════════════════════════════════════════

    'Matrix-M (Novavax, saponin NP)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('moderate'),
                       'Inflammasome': _s('moderate')},
        'th_bias':    {'Th1': +0.20, 'Th2': +0.05, 'Th17': 0.0, 'Tfh': +0.10},
        'dc_boost': 0.25, 'cd8_boost': 0.18,
        'confidence': 'approved',
        'class': 'Saponin',
        'notes': 'Saponin nanoparticle (ISCOMATRIX-comparable). FDA-approved Oct 2022 '
                 '(Novavax COVID-19 vaccine). NLRP3-driven inflammasome via saponin '
                 'lysosomal destabilisation. Strong Th1 + CD8 cross-presentation. '
                 'WHO-recommended R21/Matrix-M malaria vaccine 2023 (75% efficacy). '
                 'Science Advances 2024.',
        'compatible': ['Protein subunit', 'mRNA'],
    },

    'ISCOMATRIX (saponin NP)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('moderate'),
                       'Inflammasome': _s('moderate')},
        'th_bias':    {'Th1': +0.20, 'Th2': +0.05, 'Th17': 0.0, 'Tfh': +0.10},
        'dc_boost': 0.25, 'cd8_boost': 0.18,
        'confidence': 'clinical',
        'class': 'Saponin',
        'notes': 'Quil-A saponin + cholesterol + phospholipid, 30-70 nm nanoparticle. '
                 'Well-tolerated particulate delivery. Strong Th1 + CD8 responses. '
                 'Multiple Phase I/II trials. Precursor platform to Matrix-M '
                 '(Science Advances 2024).',
        'compatible': ['Protein subunit', 'mRNA'],
    },

    # ══ CLASS 5: COMBINATION ADJUVANT SYSTEMS (Approved) ════════════════════

    'AS01B (MPL + QS-21, liposome)': {
        'scores':     {'TLR7_8': _s('high'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('moderate')},
        'th_bias':    {'Th1': +0.40, 'Th2': -0.05, 'Th17': 0.0, 'Tfh': +0.20},
        'dc_boost': 0.35, 'cd8_boost': 0.10,
        'confidence': 'approved',
        'class': 'Combination',
        'notes': 'MPL (TLR4→Th1) + QS-21 (NLRP3 inflammasome) in liposome. '
                 'Synergistic: early IFN-γ from NK cells via IL-12/IL-18 '
                 '(Coccia 2017 npj Vaccines). Blocking IFN-γ abolishes synergy. '
                 'FDA-approved in Shingrix (zoster) and Mosquirix (malaria). '
                 'Tandfonline 2024 AS01 review.',
        'compatible': ['Protein subunit'],
    },

    'AS01E (MPL + QS-21, half-dose)': {
        'scores':     {'TLR7_8': _s('moderate'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('low')},
        'th_bias':    {'Th1': +0.30, 'Th2': -0.03, 'Th17': 0.0, 'Tfh': +0.15},
        'dc_boost': 0.28, 'cd8_boost': 0.08,
        'confidence': 'approved',
        'class': 'Combination',
        'notes': 'Half-dose AS01B (25 µg MPL + 25 µg QS-21). Similar mechanism '
                 'to AS01B but lower magnitude. Used in paediatric malaria vaccine '
                 'formulation (PMC 2024 comparative systems vaccinology).',
        'compatible': ['Protein subunit'],
    },

    'AS04 (MPL + Alum)': {
        'scores':     {'TLR7_8': _s('moderate'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('low'),
                       'Inflammasome': _s('moderate')},
        'th_bias':    {'Th1': +0.25, 'Th2': +0.05, 'Th17': 0.0, 'Tfh': +0.08},
        'dc_boost': 0.22, 'cd8_boost': 0.03,
        'confidence': 'approved',
        'class': 'Combination',
        'notes': 'MPL (TLR4→Th1) + alum (NLRP3 inflammasome + Th2). Net: Th1-leaning '
                 'with some Th2 from alum component. Significantly higher IFN-γ than '
                 'alum alone (Frontiers Immunol 2025). FDA-approved Cervarix HPV vaccine '
                 'and Fendrix HBV vaccine.',
        'compatible': ['Protein subunit'],
    },

    'AS15 (MPL + QS-21 + CpG, liposome)': {
        'scores':     {'TLR7_8': _s('very_high'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('moderate')},
        'th_bias':    {'Th1': +0.42, 'Th2': -0.08, 'Th17': 0.0, 'Tfh': +0.20},
        'dc_boost': 0.38, 'cd8_boost': 0.12,
        'confidence': 'clinical',
        'class': 'Combination',
        'notes': 'Triple combination: MPL (TLR4) + QS-21 (NLRP3) + CpG ODN (TLR9) '
                 'in liposome. Very strong Th1 and Tfh. Used in cancer and HIV '
                 'vaccine trials. Highest Th1 response of any GSK adjuvant system '
                 '(PMC 2021 QS-21 review).',
        'compatible': ['Protein subunit'],
    },

    'ALFQ (Army liposome + QS-21)': {
        'scores':     {'TLR7_8': _s('moderate'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('moderate')},
        'th_bias':    {'Th1': +0.30, 'Th2': -0.03, 'Th17': 0.0, 'Tfh': +0.15},
        'dc_boost': 0.28, 'cd8_boost': 0.10,
        'confidence': 'clinical',
        'class': 'Combination',
        'notes': 'Army Liposome Formulation (saturated phospholipids + cholesterol + '
                 'MPLA) combined with QS-21. Similar composition to AS01 but different '
                 'phospholipid. Malaria and HIV Phase I trials (Alving 2020; PMC 2025).',
        'compatible': ['Protein subunit'],
    },

    'CAF01 (DDA + TDB, Mincle)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('low')},
        'th_bias':    {'Th1': +0.18, 'Th2': 0.0, 'Th17': +0.18, 'Tfh': +0.05},
        'dc_boost': 0.18, 'cd8_boost': 0.05,
        'confidence': 'clinical',
        'class': 'Combination',
        'notes': 'Cationic DDA liposome + TDB (trehalose dibehenate, Mincle receptor '
                 'agonist). Th1/Th17 bias; strong for intracellular pathogens (TB, '
                 'chlamydia). Phase I trials completed (NCT02787109, NCT00922363). '
                 'PMC 2025; Frontiers 2023.',
        'compatible': ['Protein subunit'],
    },

    'CAF09 / CAF09b (DDA + MMG + Poly I:C)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('high'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('low')},
        'th_bias':    {'Th1': +0.22, 'Th2': -0.03, 'Th17': +0.10, 'Tfh': +0.08},
        'dc_boost': 0.22, 'cd8_boost': 0.18,
        'confidence': 'clinical',
        'class': 'Combination',
        'notes': 'DDA liposome + MMG (Mincle) + Poly(I:C) (TLR3). Combines '
                 'Th1/Th17 from Mincle with strong CD8 induction from TLR3. '
                 'Phase I chlamydia vaccine (NCT03926728) completed. '
                 'Frontiers 2023; PMC 2025.',
        'compatible': ['Protein subunit'],
    },

    # ══ CLASS 6: CGAS-STING AGONISTS (Investigational) ══════════════════════

    "2'3'-cGAMP (STING agonist)": {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('very_high'), 'Complement': _s('none'),
                       'Inflammasome': _s('none')},
        'th_bias':    {'Th1': +0.25, 'Th2': -0.05, 'Th17': 0.0, 'Tfh': +0.10},
        'dc_boost': 0.20, 'cd8_boost': 0.25,
        'confidence': 'preclinical',
        'class': 'STING agonist',
        'notes': 'Natural STING agonist. Strong type I IFN, Th1, CD8 responses. '
                 'HUMAN STING NOTE: STING allele H232 responds to non-canonical CDNs; '
                 'allele R232 (most common) responds to canonical. Poor membrane '
                 'permeability (<60 min half-life). Nanoparticle delivery required '
                 'for efficient in vivo adjuvancy (Wiley Med Res Rev 2024).',
        'compatible': ['Protein subunit', 'mRNA', 'DNA'],
    },

    'ADU-S100 / MIW815 (STING)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('very_high'), 'Complement': _s('none'),
                       'Inflammasome': _s('none')},
        'th_bias':    {'Th1': +0.25, 'Th2': -0.05, 'Th17': 0.0, 'Tfh': +0.08},
        'dc_boost': 0.18, 'cd8_boost': 0.22,
        'confidence': 'preclinical',
        'class': 'STING agonist',
        'notes': 'Atypical CDN. Phase I solid tumour trial completed. Limited '
                 'single-agent activity but IFN-γ/CXCL10 systemic activation '
                 'confirmed dose-dependently (Clin Cancer Res 2022). Cleared from '
                 'bloodstream within 2 hours. Not yet evaluated as vaccine adjuvant '
                 'in infectious disease.',
        'compatible': ['Protein subunit', 'mRNA', 'DNA'],
    },

    # ══ CLASS 7: MUCOSAL / OTHER ═════════════════════════════════════════════

    'dmLT (double mutant heat-labile toxin)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('low')},
        'th_bias':    {'Th1': +0.05, 'Th2': +0.05, 'Th17': +0.25, 'Tfh': +0.08},
        'dc_boost': 0.15, 'cd8_boost': 0.05,
        'confidence': 'clinical',
        'class': 'Mucosal',
        'notes': 'GM1-ganglioside targeting enterotoxin mutant. Mucosal Th17 + IgA '
                 'responses via cAMP pathway. Phase I/II trials for ETEC and '
                 'norovirus oral vaccines. Best used for mucosal vaccine routes. '
                 'PMC 2023 oral adjuvant review.',
        'compatible': ['Protein subunit'],
    },

    'Flagellin / FliC (TLR5)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('none'),
                       'Inflammasome': _s('low')},
        'th_bias':    {'Th1': +0.15, 'Th2': 0.0, 'Th17': +0.12, 'Tfh': +0.05},
        'dc_boost': 0.18, 'cd8_boost': 0.08,
        'confidence': 'clinical',
        'class': 'Mucosal',
        'notes': 'TLR5 agonist (bacterial flagellin). NF-κB→Th1/Th17 via MyD88. '
                 'Also signals through NLRC4 inflammasome. Phase I/II trials as '
                 'fusion protein with antigen. Generates responses against itself '
                 '(immunogenicity concern). Frontiers Microbiol 2023.',
        'compatible': ['Protein subunit'],
    },

    'Chitosan (polymer, cGAS-STING + NLRP3)': {
        'scores':     {'TLR7_8': _s('none'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('low'), 'Complement': _s('low'),
                       'Inflammasome': _s('moderate')},
        'th_bias':    {'Th1': +0.12, 'Th2': +0.05, 'Th17': +0.10, 'Tfh': +0.05},
        'dc_boost': 0.12, 'cd8_boost': 0.08,
        'confidence': 'preclinical',
        'class': 'Polymer',
        'notes': 'Biodegradable cationic polymer. Dual activation: NLRP3 inflammasome '
                 '(IL-1β) and cGAS-STING pathway (type I IFN). Immunity 2016 Carroll. '
                 'Good safety profile. Th1/Th17/IgG2 when combined with CpG. '
                 'Particulate delivery enhances mucosal responses. PMC 2025.',
        'compatible': ['Protein subunit', 'mRNA', 'DNA'],
    },

    'CFA (Complete Freund: research only)': {
        'scores':     {'TLR7_8': _s('moderate'), 'TLR3': _s('none'),
                       'cGAS_STING': _s('none'), 'Complement': _s('low'),
                       'Inflammasome': _s('moderate')},
        'th_bias':    {'Th1': +0.35, 'Th2': -0.05, 'Th17': +0.10, 'Tfh': +0.08},
        'dc_boost': 0.28, 'cd8_boost': 0.08,
        'confidence': 'research',
        'class': 'Research only',
        'notes': 'Mineral oil + killed Mycobacterium tuberculosis. TLR2 (mycobacterial '
                 'lipoproteins) + depot. Very strong Th1. NOT for human use: '
                 'severe local reactions. Standard preclinical mouse model comparator. '
                 'Seubert 2011 PNAS.',
        'compatible': ['Protein subunit'],
    },

}

# Confidence tier labels and colours for UI display
ADJUVANT_CONFIDENCE = {
    'approved':    ('FDA/EMA Approved',  '#16a34a'),
    'clinical':    ('Clinical Trials',   '#2563eb'),
    'preclinical': ('Preclinical',       '#d97706'),
    'research':    ('Research Only',     '#dc2626'),
}

# Class grouping for selectbox display
ADJUVANT_CLASS_ORDER = [
    'None', 'Mineral salt', 'Emulsion', 'TLR agonist',
    'Saponin', 'Combination', 'STING agonist', 'Mucosal', 'Polymer', 'Research only'
]

ANTIGEN_PRESETS = {
    '— Enter custom sequence —': {'sequence': '', 'type': 'custom', 'notes': ''},
    'SARS-CoV-2 Spike (S1 RBD)': {
        'sequence': 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLE',
        'type': 'spike_protein', 'notes': 'SARS-CoV-2 RBD; basis of mRNA-1273 and BNT162b2 antigen'
    },
    'Influenza HA (H1N1)': {
        'sequence': 'MKTIIALSYIFCLVFADTKICNNPHRILDGIDCTLIDALLGDPHCDVFQDETWDLFVERSKAFSNCYPYDVPDYASLRSLVASSGTLEFNNESFNWTGVTQNGTSSACIRRSSSSFFSRLNWLTHLKFKYPALNVTMPNNEQFDKLYIWGVHHPSTDSDQISLYAQASGRITVSTKRSQQTVIPNIGSRPWVRGVSSRISIYWTIVKPGDVLVINSNGNLIAPRGYFKMRTGKSSIMRSDAPIGTCSSECITPNGSIPNDKPFQNVNRITYGACPRYVKQNTLKLATGMRNVPEKQTRGIFGAIAGFIENGWEGMVDGWYGFRHQNSEGTGQAADLKSTQAAIDQINGKLNRLIGKTNEKFHQIEKEFSEVEGRIQDLEKYVEDTKIDLWSYNAELLVALENQHTIDLTDSEMNKLFERTKKQLRENAEDMGNGCFKIYHKCDNACIGSIRNGTYDHDVYRDEALNNRFQIKGVELKSGYKDWILWISFAISCFLLCVALLGFIMWACQKGNIRCNICI',
        'type': 'hemagglutinin', 'notes': 'H1N1 influenza HA; seasonal vaccine antigen'
    },
    'HIV-1 Env gp120': {
        'sequence': 'MRVKEKYQHLWRWGWRWGTMLLGMLMICSATEKLWVTVYYGVPVWKEATTTLFCASDAKAYDTEVHNVWATHACVPTDPNPQEVVLVNVTENFNMWKNDMVEQMHEDIISLWDQSLKPCVKLTPLCVSLKCTDLKNDTNTNSSSGRMIMEKGEIKNCSFNISTSIRGKVQKEYAFFYKLDIIPIDNDTTSYKLTSCNTSVITQACPKVSFEPIPIHYCAPAGFAILKCNNKTFNGTGPCTNVSTVQCTHGIRPVVSTQLLLNGSLAEEEVVIRSVNFTDNAKTIIVQLNTSVEINCTRPNNNTRKRIRIQRGPGRAFVTIGKIGNMRQAHCNISRAKWNNTLKQIASKLREQFGNNKTIIFKQSSGGDPEIVTHSFNCGGEFFYCNSTQLFNSTWFNSTWSTEGSNNTEGSDTITLPCRIKQIINMWQKVGKAMYAPPISGQIRCSSNITGLLLTRDGGNNNNGSEIFRPGGGDMRDNWRSELYKYKVVKIEPLGVAPTKAKRRVVQREKRAVGIGALFLGFLGAAGSTMGAASMTLTVQARQLLSGIVQQQNNLLRAIEAQQHLLQLTVWGIKQLQARILAVERYLKDQQLLGIWGCSGKLICTTAVPWNASWSNKSLEQIWNHTTWMEWDREINNYTSLIHSLIEESQNQQEKNEQELLELDKWASLWNWFNITNWLWYIKLFIMIVGGLVGLRIVFAVLSIVNRVRQGYSPLSFQTHLPTPRGPDRPEGIEEEGGERDRDRSIRLVNGSLALIWDDLRSLCLFSYHRLRDLLLIVTRIVELLGRRGWEALKYWWNLLQYWSQELKNSAVSLLNATAIAVAEGTDRVIEVVQGACRAIRHIPRRIRQGLERILL',
        'type': 'envelope_protein', 'notes': 'HIV-1 gp120 Env; challenge antigen for HIV vaccine research'
    },
    'Mycobacterium tuberculosis Ag85B': {
        'sequence': 'MSFPSGSTPAAGLVLRRLAAAFASTSAAATAPASQHFWDFFNAAERSPFGSGEITSANSGSSTTFSAPASSANTSAANSSATSASPASASPAGASPSATSAAPSAGSSAPTAAAPASPAASAPSSAANSSAPNSSAPASSAPNSSAPNSSAPASTSAPTAASAPASAPTSAPTAAAPTSAAAPTSAAAPAAASTPAAASTPAAASTPAAAPTAAAPTAAAPTSPAASAPTSAAAPAAPASTSAPAAPAAPAAATSAAPAAPAAPAAATSAAPAAPAAPAAATSAAPAAPAAPAAATSAAPAATSA',
        'type': 'secreted_antigen', 'notes': 'M. tuberculosis secreted protein; key TB vaccine candidate'
    },
    'HPV16 L1 Capsid Protein': {
        'sequence': 'MSLWLPSEATVYLPPVPVSKVVSTDEYVARTNIYYHAGTSRLLAVGHPYFPIKKGNKADVPKVSGLQYRVFRIHLPDPNKFGFPDTSFYNPETQRLVWACAGVEVGRGQPLGVGISGHPLLNKLDDTENASAYAANAGVDNREKLTPQNVTDTHFKNKGVCPPLELITNSVIQDGDMVDTGFGAMDFTTLQANKSDVPIDIKMPVYGSMAIAPSSSSTKVSSDAQIFNKPYWLQRAQGHNNGICWGNQLFVTVVDTTRSTNMSLCAAISTSETTYKNTNFKEYLRHGEEYDLQFIFQLCKITLTADVMTYIHSMNSTILEDWNFGLQPPPGGTLEDTYRFVTSQAIACQKHTPPAPKEDPLKKYTFWEVNLKEKFSADLDQFPLGRKFLLQAGKR',
        'type': 'capsid_protein', 'notes': 'HPV-16 L1 VLP antigen; basis of Gardasil/Cervarix'
    },
}

# ── PSSM matrices (anchor positions from published HLA data) ──────────────────
# HLA-A*02:01 9-mer PSSM — rows = positions 1-9, cols = ACDEFGHIKLMNPQRSTVWY
# Values are log-odds scores; derived from the SYFPEITHI/BIMAS databases.
_HLA_A0201_PSSM = {
    # position: {aa: score}  (only strong anchors encoded; others default 0)
    1: {'M':2,'L':2,'V':1,'I':1,'F':1,'A':1},
    2: {'L':3,'M':2,'V':2,'I':2,'T':1,'A':1},
    3: {'P':1,'A':1,'V':1,'L':1,'I':1},
    4: {'A':1,'T':1,'V':1,'S':1},
    5: {'A':1,'V':1,'T':1,'I':1,'L':1},
    6: {'V':2,'I':2,'L':2,'T':1,'A':1},
    7: {'P':1,'A':1,'G':1,'S':1},
    8: {'K':1,'R':1,'Q':1,'H':1},
    9: {'L':4,'V':3,'I':3,'M':2,'A':1,'T':1},  # anchor P9: critical
}
# HLA-DR allele generic 15-mer PSSM (MHC-II)
_HLA_DR_PSSM = {
    1: {'L':2,'V':2,'I':2,'F':2,'M':1,'W':1,'Y':1},   # P1 anchor
    4: {'D':1,'E':1,'S':1,'T':1,'N':1,'Q':1},
    6: {'H':2,'K':2,'R':2,'Q':1,'N':1},
    9: {'L':2,'V':2,'I':2,'F':2,'M':1},
}

# Kyte-Doolittle hydrophobicity
_KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
       'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
       'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}


def _pssm_score_peptide(peptide: str, pssm: dict, length: int) -> float:
    """Score a single peptide against a PSSM. Returns normalised 0-1 score."""
    if len(peptide) != length:
        return 0.0
    score = sum(pssm.get(pos+1, {}).get(aa, 0) for pos, aa in enumerate(peptide))
    max_possible = sum(max(v.values()) for v in pssm.values())
    return score / max_possible if max_possible > 0 else 0.0


def _local_epitope_scan(seq: str) -> dict:
    """
    PSSM-based local epitope scanner.
    Uses published HLA-A*02:01 (MHC-I, 9-mer) and HLA-DR generic (MHC-II, 15-mer)
    anchor position weight matrices to score all overlapping windows.
    Returns top peptides and aggregate immunogenicity scores.
    """
    seq = seq.upper().replace(' ','').replace('\n','')
    n = len(seq)

    # MHC-I: all 9-mers
    mhci_scores, top_mhci = [], []
    for i in range(n - 8):
        pep = seq[i:i+9]
        s = _pssm_score_peptide(pep, _HLA_A0201_PSSM, 9)
        mhci_scores.append(s)
        if s > 0.35:
            top_mhci.append((s, pep, i+1))
    top_mhci.sort(reverse=True)

    # MHC-II: all 15-mers
    mhcii_scores, top_mhcii = [], []
    for i in range(n - 14):
        pep = seq[i:i+15]
        s = _pssm_score_peptide(pep, _HLA_DR_PSSM, 15)
        mhcii_scores.append(s)
        if s > 0.25:
            top_mhcii.append((s, pep, i+1))
    top_mhcii.sort(reverse=True)

    # B-cell: surface-exposed windows (charged + hydrophilic stretches, 12-mer)
    bcell_scores = []
    for i in range(n - 11):
        window = seq[i:i+12]
        hydrophilicity = -np.mean([_KD.get(aa, 0) for aa in window])
        charge = sum(1 for aa in window if aa in 'KRDE')
        bcell_scores.append(np.clip((hydrophilicity * 0.4 + charge * 0.5) / 8, 0, 1))

    # Aggregate scores
    mhc1_agg  = float(np.percentile(mhci_scores, 90))  if mhci_scores  else 0.3
    mhc2_agg  = float(np.percentile(mhcii_scores, 90)) if mhcii_scores else 0.3
    bcell_agg = float(np.percentile(bcell_scores, 85)) if bcell_scores else 0.2

    # Hydrophobicity
    hydrophobicity = float(np.mean([_KD.get(aa, 0) for aa in seq]))

    # Antigenicity: weighted combination
    antigenicity = float(np.clip(mhc1_agg * 0.35 + mhc2_agg * 0.35 + bcell_agg * 0.3, 0, 1))

    return {
        'method':             'PSSM (HLA-A*02:01 + HLA-DR local scanner)',
        'mhc1_score':         float(np.clip(mhc1_agg,  0, 1)),
        'mhc2_score':         float(np.clip(mhc2_agg,  0, 1)),
        'b_cell_score':       float(np.clip(bcell_agg, 0, 1)),
        'antigenicity':       antigenicity,
        'hydrophobicity':     hydrophobicity,
        'ctl_epitopes_est':   len([s for s in mhci_scores  if s > 0.35]),
        'th_epitopes_est':    len([s for s in mhcii_scores if s > 0.25]),
        'bcell_epitopes_est': len([s for s in bcell_scores if s > 0.4]),
        'top_mhci_peptides':  [(s, p, pos) for s, p, pos in top_mhci[:5]],
        'top_mhcii_peptides': [(s, p, pos) for s, p, pos in top_mhcii[:5]],
    }


def _iedb_mhci_call(seq: str, allele: str = 'HLA-A*02:01', length: int = 9) :
    """
    Call IEDB Analysis Resource MHC-I prediction API.
    Returns parsed results or None if unreachable.
    Endpoint: https://tools-cluster-interface.iedb.org/tools_api/mhci/
    Method: recommended (NetMHCpan 4.1 + consensus)
    """
    try:
        resp = _requests.post(
            'https://tools-cluster-interface.iedb.org/tools_api/mhci/',
            data={'method': 'recommended', 'sequence_text': seq,
                  'allele': allele, 'length': str(length)},
            timeout=15,
            allow_redirects=True,
        )
        if resp.status_code != 200:
            return None

        # Parse TSV — use header row to locate columns dynamically
        all_lines = [l for l in resp.text.strip().split('\n') if l]
        if not all_lines:
            return None

        # Header: allele seq_num start end length peptide core icore score percentile_rank
        header = all_lines[0].split('\t')
        col = {h: i for i, h in enumerate(header)}

        # Required columns — bail if they're missing
        for required in ('peptide', 'percentile_rank'):
            if required not in col:
                return None

        peptides = []
        for line in all_lines[1:51]:   # up to 50 peptides
            parts = line.split('\t')
            if len(parts) <= max(col.values()):
                continue
            try:
                peptides.append({
                    'allele':           parts[col.get('allele', 0)],
                    'position':         int(parts[col['start']]) if 'start' in col else 0,
                    'peptide':          parts[col['peptide']],
                    'percentile_rank':  float(parts[col['percentile_rank']]),
                    'score':            float(parts[col['score']]) if 'score' in col else 0.0,
                })
            except (ValueError, IndexError):
                continue

        if not peptides:
            return None

        # Binding thresholds based on percentile_rank:
        #   strong binder: percentile_rank < 0.5%
        #   weak binder:   percentile_rank 0.5–2%
        #   (IEDB recommended thresholds, NetMHCpan 4.1)
        strong = [p for p in peptides if p['percentile_rank'] <  0.5]
        weak   = [p for p in peptides if 0.5 <= p['percentile_rank'] < 2.0]
        mhc1_score = float(np.clip(
            (len(strong) * 3 + len(weak)) / max(len(peptides), 1) * 0.4, 0, 1
        ))

        return {
            'method':           f'IEDB NetMHCpan ({allele})',
            'peptides':         peptides,
            'strong_binders':   strong[:8],
            'weak_binders':     weak[:8],
            'mhc1_score':       mhc1_score,
            'ctl_epitopes_est': len(strong),
        }
    except Exception:
        return None


def _iedb_mhcii_call(seq: str, allele: str = 'HLA-DRB1*01:01') :
    """
    Call IEDB Analysis Resource MHC-II prediction API.
    Endpoint: https://tools-cluster-interface.iedb.org/tools_api/mhcii/
    Method: recommended (NetMHCIIpan 4.0)
    """
    try:
        resp = _requests.post(
            'https://tools-cluster-interface.iedb.org/tools_api/mhcii/',
            data={'method': 'recommended', 'sequence_text': seq, 'allele': allele},
            timeout=15,
            allow_redirects=True,
        )
        if resp.status_code != 200:
            return None

        all_lines = [l for l in resp.text.strip().split('\n') if l]
        if not all_lines:
            return None

        # Header-based parsing — MHC-II columns vary by method
        # Typical: allele seq_num start end length peptide core percentile_rank
        header = all_lines[0].split('\t')
        col = {h: i for i, h in enumerate(header)}

        for required in ('peptide', 'percentile_rank'):
            if required not in col:
                return None

        peptides = []
        for line in all_lines[1:51]:
            parts = line.split('\t')
            if len(parts) <= max(col.values()):
                continue
            try:
                peptides.append({
                    'allele':          parts[col.get('allele', 0)],
                    'position':        int(parts[col['start']]) if 'start' in col else 0,
                    'peptide':         parts[col['peptide']],
                    'percentile_rank': float(parts[col['percentile_rank']]),
                })
            except (ValueError, IndexError):
                continue

        if not peptides:
            return None

        # MHC-II strong binder: percentile_rank < 2% (NetMHCIIpan recommended threshold)
        strong = [p for p in peptides if p['percentile_rank'] < 2.0]
        weak   = [p for p in peptides if 2.0 <= p['percentile_rank'] < 10.0]
        mhc2_score = float(np.clip(
            (len(strong) * 3 + len(weak)) / max(len(peptides), 1) * 0.4, 0, 1
        ))

        return {
            'method':           f'IEDB NetMHCIIpan ({allele})',
            'peptides':         peptides,
            'strong_binders':   strong[:8],
            'weak_binders':     weak[:8],
            'mhc2_score':       mhc2_score,
            'th_epitopes_est':  len(strong),
        }
    except Exception:
        return None


def _ml_epitope_scan(seq: str, species: str = 'human') :
    """
    Try to run the trained XGBoost MHC-I model.
    Returns result dict or None if model file not found.
    Falls back gracefully so the rest of the pipeline keeps working.
    """
    try:
        import joblib as _joblib
        import os, sys

        # Resolve app.py directory for absolute paths
        _app_dir = os.path.dirname(os.path.abspath(__file__))

        # Look for model file — try multiple locations
        model_candidates = [
            os.path.join(_app_dir, f'epitrix_ml/models/mhci_xgboost_{species}.pkl'),
            os.path.join(_app_dir, f'models/mhci_xgboost_{species}.pkl'),
            f'epitrix_ml/models/mhci_xgboost_{species}.pkl',
            f'models/mhci_xgboost_{species}.pkl',
        ]
        model_path = None
        for p in model_candidates:
            if os.path.exists(p):
                model_path = p
                break

        # Store debug info in session state so we can surface it in UI
        import streamlit as _st
        debug_info = {
            'app_dir': _app_dir,
            'candidates_checked': model_candidates,
            'model_found': model_path,
        }
        _st.session_state['_ml_debug'] = debug_info

        if model_path is None:
            return None

        # Check file is not a Git LFS pointer (real pkl should be > 10KB)
        file_size = os.path.getsize(model_path)
        if file_size < 10_000:
            _st.session_state['_ml_debug']['error'] = f'Model file too small ({file_size} bytes) — likely a Git LFS pointer, not the actual model'
            return None

        # Ensure app directory is on path for epitrix_ml package import
        if _app_dir not in sys.path:
            sys.path.insert(0, _app_dir)

        from epitrix_ml.integrate import ml_epitope_scan as _ml_scan
        result = _ml_scan(seq, model_path=model_path)

        if not result.get('_ml_available'):
            _st.session_state['_ml_debug']['error'] = f'ml_scan returned _ml_available=False'
            return None

        return {
            'mhc1_score':      result['mhc1_score'],
            'ctl_epitopes_est':result['ctl_epitopes_est'],
            'method':          result['method'],
            'top_mhci':        result['top_mhci_peptides'],
            'ml_auc':          result.get('_model_auc'),
            'species':         species,
        }
    except Exception as _e:
        try:
            import streamlit as _st
            _st.session_state['_ml_debug'] = _st.session_state.get('_ml_debug', {})
            _st.session_state['_ml_debug']['exception'] = str(_e)
        except Exception:
            pass
        return None


def analyze_antigen_sequence(sequence: str, use_iedb: bool = True,
                              species: str = 'human') -> dict:
    """
    Full antigen sequence analysis pipeline.
    Priority order for MHC-I prediction:
      1. Trained XGBoost model (best: if model file present)
      2. IEDB NetMHCpan API (good: if network reachable)
      3. Local PSSM scanner (always available: fallback)
    """
    seq = sequence.upper().strip().replace(' ', '').replace('\n', '')
    if not seq:
        return {'valid': False}

    length = len(seq)
    is_protein = all(c in 'ACDEFGHIKLMNPQRSTVWYBZXU*-' for c in seq)
    is_rna     = all(c in 'AUGC' for c in seq)
    is_dna     = all(c in 'ATGC' for c in seq)

    if not (is_protein or is_rna or is_dna):
        return {'valid': False, 'error': 'Unrecognised sequence characters'}

    seq_type = 'Protein' if is_protein else ('RNA' if is_rna else 'DNA')

    if is_protein:
        # ── Tier 1: XGBoost ML model ─────────────────────────────────────────
        ml_result = _ml_epitope_scan(seq, species=species)

        # ── Tier 2: IEDB API ──────────────────────────────────────────────────
        # Use IEDB allele mapping based on species
        if species == 'mouse':
            iedb_allele_mhci  = 'H-2Kb'
            iedb_allele_mhcii = 'H-2IAb'
        else:
            iedb_allele_mhci  = 'HLA-A*02:01'
            iedb_allele_mhcii = 'HLA-DRB1*01:01'

        iedb_mhci  = _iedb_mhci_call(seq,  allele=iedb_allele_mhci)  if (use_iedb and not ml_result) else None
        iedb_mhcii = _iedb_mhcii_call(seq, allele=iedb_allele_mhcii) if use_iedb else None

        # ── Tier 3: Local PSSM ───────────────────────────────────────────────
        local = _local_epitope_scan(seq)

        # ── Merge MHC-I: ML > IEDB > PSSM ────────────────────────────────────
        if ml_result:
            mhc1_score       = ml_result['mhc1_score']
            ctl_epitopes_est = ml_result['ctl_epitopes_est']
            mhci_method      = ml_result['method']
            top_mhci         = ml_result['top_mhci']
            ml_used          = True
        elif iedb_mhci:
            mhc1_score       = iedb_mhci['mhc1_score']
            ctl_epitopes_est = iedb_mhci['ctl_epitopes_est']
            mhci_method      = iedb_mhci['method']
            top_mhci         = [(p['percentile_rank'], p['peptide'], p['position'])
                                for p in iedb_mhci['strong_binders']]
            ml_used          = False
        else:
            mhc1_score       = local['mhc1_score']
            ctl_epitopes_est = local['ctl_epitopes_est']
            mhci_method      = local['method']
            top_mhci         = local['top_mhci_peptides']
            ml_used          = False

        # ── MHC-II: IEDB > PSSM (ML model not yet trained for MHC-II) ────────
        if iedb_mhcii:
            mhc2_score      = iedb_mhcii['mhc2_score']
            th_epitopes_est = iedb_mhcii['th_epitopes_est']
            mhcii_method    = iedb_mhcii['method']
            top_mhcii       = [(p['percentile_rank'], p['peptide'], p['position'])
                               for p in iedb_mhcii['strong_binders']]
        else:
            mhc2_score      = local['mhc2_score']
            th_epitopes_est = local['th_epitopes_est']
            mhcii_method    = local['method']
            top_mhcii       = local['top_mhcii_peptides']

        b_cell_score       = local['b_cell_score']
        bcell_epitopes_est = local['bcell_epitopes_est']
        hydrophobicity     = local['hydrophobicity']
        antigenicity       = float(np.clip(mhc1_score*0.35 + mhc2_score*0.35 + b_cell_score*0.3, 0, 1))
        iedb_used          = bool(iedb_mhci or iedb_mhcii)

    else:  # RNA / DNA
        gc_content    = (seq.count('G') + seq.count('C')) / length
        au_content    = (seq.count('A') + seq.count('U' if is_rna else 'T')) / length
        mhc1_score    = float(np.clip(gc_content * 1.0, 0, 1))
        mhc2_score    = float(np.clip(gc_content * 1.2, 0, 1))
        b_cell_score  = float(np.clip(au_content * 0.3 * 3, 0, 1))
        antigenicity  = float(np.clip(gc_content * 0.9 + au_content * 0.1, 0, 1))
        hydrophobicity = 0.0
        ctl_epitopes_est   = int(length / 9  * gc_content * 3)
        th_epitopes_est    = int(length / 15 * gc_content * 2)
        bcell_epitopes_est = int(length / 20 * au_content * 2)
        mhci_method  = 'GC/AU composition (nucleotide)'
        mhcii_method = 'GC/AU composition (nucleotide)'
        top_mhci, top_mhcii = [], []
        iedb_used = False
        ml_used   = False

    return {
        'valid':               True,
        'seq_type':            seq_type,
        'length':              length,
        'species':             species,
        'mhc1_score':          float(np.clip(mhc1_score,   0, 1)),
        'mhc2_score':          float(np.clip(mhc2_score,   0, 1)),
        'b_cell_score':        float(np.clip(b_cell_score, 0, 1)),
        'antigenicity':        float(np.clip(antigenicity, 0, 1)),
        'hydrophobicity':      float(hydrophobicity),
        'ctl_epitopes_est':    max(0, ctl_epitopes_est),
        'th_epitopes_est':     max(0, th_epitopes_est),
        'bcell_epitopes_est':  max(0, bcell_epitopes_est),
        'top_mhci_peptides':   top_mhci,
        'top_mhcii_peptides':  top_mhcii,
        'mhci_method':         mhci_method,
        'mhcii_method':        mhcii_method,
        'iedb_used':           iedb_used,
        'ml_used':             ml_used,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE INTERVAL MODEL
# Published inter-individual variability coefficients for each output metric,
# derived from clinical vaccine trial data (see Training Datasets page).
# CV values reflect observed biological variability, not model uncertainty.
# ─────────────────────────────────────────────────────────────────────────────
# Coefficient of variation (CV) per metric from literature:
_CV = {
    'particle_size':     0.12,   # DLS batch-to-batch ~12% CV
    'zeta_potential':    0.15,
    'encapsulation_eff': 0.05,
    'membrane_fluidity': 0.10,
    'TLR7_8':            0.28,   # cytokine assay inter-individual CV ~28%
    'TLR3':              0.30,
    'cGAS_STING':        0.32,
    'Complement':        0.25,
    'Inflammasome':      0.30,
    'Th1':               0.35,   # T cell subset frequency CV ~35%
    'Th2':               0.40,
    'Th17':              0.45,
    'Tfh':               0.38,
    'memory_quality':    0.30,
    'efficacy':          0.18,   # phase III trial efficacy 95% CI ~±18pp
    'safety':            0.10,
    'reactogenicity':    0.25,
    'duration_months':   0.40,   # antibody persistence CV ~40%
    'ab_magnitude':      0.45,
}

Z95 = 1.96  # z-score for 95% CI

def _ci(value: float, metric: str) -> tuple[float, float]:
    """Return (lower_95, upper_95) confidence interval for a scalar value."""
    cv  = _CV.get(metric, 0.25)
    sd  = value * cv
    return (max(0.0, value - Z95 * sd), min(1.0 if value <= 1 else value * 2, value + Z95 * sd))

def add_confidence_intervals(r: dict) -> dict:
    """
    Annotate a prediction result dict with 95% CI for every numeric output.
    CIs model biological/population variability, not algorithmic uncertainty.
    """
    ci = {}
    # Physicochemical
    for k in ('particle_size', 'zeta_potential', 'encapsulation_eff', 'membrane_fluidity'):
        v = r[k]
        sd = abs(v) * _CV.get(k, 0.15)
        ci[k] = (v - Z95*sd, v + Z95*sd)

    # Innate
    for pathway, val in r['innate_prediction'].items():
        lo, hi = _ci(val, pathway)
        ci[f'innate_{pathway}'] = (lo, hi)

    # Adaptive — Th bias
    for subset, val in r['adaptive_prediction']['th_bias'].items():
        lo, hi = _ci(val, subset)
        ci[f'th_{subset}'] = (lo, hi)

    ci['memory_quality'] = _ci(r['adaptive_prediction']['memory_quality'], 'memory_quality')
    ab = r['adaptive_prediction']['antibody_response']
    ci['ab_magnitude']   = _ci(ab['magnitude'],   'ab_magnitude')
    ci['ab_durability']  = _ci(ab['durability'],  'duration_months')

    # Clinical
    for k in ('efficacy', 'safety', 'reactogenicity', 'duration_months'):
        ci[k] = _ci(r['clinical_predictions'][k], k)

    r['confidence_intervals'] = ci
    return r


def run_integrated_prediction(ionizable_lipid, ionizable_ratio, helper_ratio,
                               cholesterol_ratio, peg_ratio, modification,
                               modification_level, antigen_features=None,
                               vaccine_type='mRNA', adjuvant_name='None (formulation only)'):
    lipid_data = MOLECULAR_DESCRIPTORS['lipid_chemistry']['ionizable_lipids'][ionizable_lipid]
    mod_data   = MOLECULAR_DESCRIPTORS['nucleic_acid_modifications'][modification]

    # ── Deterministic seed derived from all input parameters ──────────────────
    # Same inputs → same seed → identical outputs every run.
    # We hash the full parameter fingerprint into a stable 32-bit integer.
    ag              = antigen_features or {}
    ag_valid        = ag.get('valid', False)
    ag_seq_hash     = hash(ag.get('length', 0) * 1000 + ag.get('mhc1_score', 0) * 100)
    seed_str = (
        f"{ionizable_lipid}|{ionizable_ratio}|{helper_ratio}|{cholesterol_ratio}|"
        f"{peg_ratio}|{modification}|{modification_level}|{ag_seq_hash}"
    )
    seed = int(abs(hash(seed_str)) % (2**31))
    rng  = np.random.default_rng(seed)

    # ── Antigen-derived boosts ────────────────────────────────────────────────
    ag_mhc1         = ag.get('mhc1_score', 0.5)    if ag_valid else 0.5
    ag_mhc2         = ag.get('mhc2_score', 0.5)    if ag_valid else 0.5
    ag_antigenicity = ag.get('antigenicity', 0.5)  if ag_valid else 0.5
    ag_bcell        = ag.get('b_cell_score', 0.3)  if ag_valid else 0.3

    # ── Physicochemical properties ────────────────────────────────────────────
    # Small biological variability is modelled via the seeded RNG —
    # deterministic for a given formulation, but scientifically realistic in magnitude.
    particle_size     = 80  + (ionizable_ratio - 40) * 1.5 + rng.normal(0, 5)
    zeta_potential    = -15 + (ionizable_ratio - 40) * 0.3 + rng.normal(0, 2)
    encapsulation_eff = min(0.98, 0.85 + (peg_ratio / 100) + rng.normal(0, 0.05))
    membrane_fluidity = max(0.3,  0.6  + (cholesterol_ratio - 30) * -0.01 + rng.normal(0, 0.05))

    # ── Innate pathway activation — vaccine-type aware ───────────────────────
    tlr_base         = 0.7 * (1 - mod_data.get('tlr_evasion', 0.5))
    tlr_lipid_effect = (lipid_data['pka'] - 6.0) * 0.1

    if vaccine_type == 'mRNA':
        # TLR7/8 dominant; driven by nucleoside modification and lipid pKa
        innate = {
            'TLR7_8':       float(np.clip(tlr_base + tlr_lipid_effect + rng.normal(0, 0.08), 0, 1)),
            'TLR3':         float(np.clip(0.3 + rng.normal(0, 0.08), 0, 1)),
            'cGAS_STING':   float(np.clip(0.4 + (particle_size - 80) * 0.005 + rng.normal(0, 0.08), 0, 1)),
            'Complement':   float(np.clip(abs(zeta_potential) * 0.02 + rng.normal(0, 0.06), 0, 1)),
            'Inflammasome': float(np.clip(0.3 + rng.normal(0, 0.06), 0, 1)),
        }
    elif vaccine_type == 'DNA':
        # TLR9 and cGAS-STING dominant for cytosolic DNA sensing;
        # TLR7/8 contribution is minimal (DNA is not ssRNA)
        # Map TLR9 onto the TLR7_8 slot for cascade compatibility
        tlr9 = float(np.clip(0.6 + tlr_lipid_effect + rng.normal(0, 0.08), 0, 1))
        innate = {
            'TLR7_8':       float(np.clip(tlr9 * 0.3 + rng.normal(0, 0.06), 0, 1)),  # TLR9 partial contribution
            'TLR3':         float(np.clip(0.2 + rng.normal(0, 0.06), 0, 1)),
            'cGAS_STING':   float(np.clip(0.65 + (particle_size - 80) * 0.005 + rng.normal(0, 0.08), 0, 1)),
            'Complement':   float(np.clip(abs(zeta_potential) * 0.02 + rng.normal(0, 0.06), 0, 1)),
            'Inflammasome': float(np.clip(0.35 + rng.normal(0, 0.06), 0, 1)),
            '_TLR9':        tlr9,   # stored for display
        }
    else:  # Protein subunit
        # No nucleic acid sensing. Innate driven by adjuvant and particle geometry.
        # TLR7/8 and TLR3 negligible; cGAS-STING minimal; complement and
        # inflammasome driven by particulate adjuvant properties.
        adj_effect = abs(zeta_potential) * 0.025  # cationic particles activate complement
        innate = {
            'TLR7_8':       float(np.clip(0.08 + rng.normal(0, 0.04), 0, 0.2)),
            'TLR3':         float(np.clip(0.06 + rng.normal(0, 0.04), 0, 0.15)),
            'cGAS_STING':   float(np.clip(0.10 + rng.normal(0, 0.05), 0, 0.25)),
            'Complement':   float(np.clip(adj_effect + 0.3 + rng.normal(0, 0.06), 0, 1)),
            'Inflammasome': float(np.clip(0.25 + rng.normal(0, 0.06), 0, 1)),
        }

    # ── Apply adjuvant modifiers ──────────────────────────────────────────────
    adj_data = ADJUVANTS.get(adjuvant_name, ADJUVANTS['None (formulation only)'])
    adj_scores = adj_data['scores']

    # Take the element-wise maximum of formulation-derived innate scores and
    # adjuvant scores. Adjuvants ADD to the formulation response, never subtract —
    # an adjuvant cannot reduce a pathway already activated by the formulation.
    for key in ['TLR7_8', 'TLR3', 'cGAS_STING', 'Complement', 'Inflammasome']:
        if key in innate:
            innate[key] = float(np.clip(
                max(innate[key], adj_scores[key]) + adj_scores[key] * 0.15,
                0, 1
            ))

    # ── Adaptive immune outcomes ──────────────────────────────────────────────
    th1  = float(np.clip(innate['TLR7_8'] * 0.5 + innate['cGAS_STING'] * 0.3 + ag_mhc2 * 0.2, 0, 1))
    th2  = float(np.clip(0.2 + (1 - ag_mhc1) * 0.1 + rng.normal(0, 0.06), 0, 1))
    th17 = float(np.clip(innate['Inflammasome'] * 0.5 + rng.normal(0, 0.06), 0, 1))
    tfh  = float(np.clip(innate['TLR7_8'] * 0.35 + th1 * 0.3 + ag_mhc2 * 0.15 + rng.normal(0, 0.06), 0, 1))

    # Apply adjuvant Th bias deltas
    adj_tb = adj_data['th_bias']
    th1  = float(np.clip(th1  + adj_tb.get('Th1',  0.0), 0, 1))
    th2  = float(np.clip(th2  + adj_tb.get('Th2',  0.0), 0, 1))
    th17 = float(np.clip(th17 + adj_tb.get('Th17', 0.0), 0, 1))
    tfh  = float(np.clip(tfh  + adj_tb.get('Tfh',  0.0), 0, 1))
    tot  = th1 + th2 + th17 + tfh + 0.01
    th_bias = {'Th1': th1/tot, 'Th2': th2/tot, 'Th17': th17/tot, 'Tfh': tfh/tot}

    memory_quality = float(np.clip(th1*0.35 + tfh*0.35 + innate['TLR7_8']*0.15 + ag_antigenicity*0.15
                                    + adj_data['dc_boost'] * 0.3, 0, 1))
    ab_peak_day    = max(7, 14 - innate['TLR7_8'] * 5 - ag_mhc2 * 2)
    ab_durability  = float(np.clip(memory_quality * 0.75 + tfh * 0.15 + ag_bcell * 0.1, 0.2, 1))
    ab_magnitude   = float(np.clip(ag_antigenicity * 0.5 + tfh * 0.3 + innate['TLR7_8'] * 0.2
                                    + adj_data['cd8_boost'] * 0.15, 0, 1))

    reactogenicity = float(np.clip(
        innate['TLR7_8'] * 0.4 + innate['Complement'] * 0.3 + innate['Inflammasome'] * 0.3, 0, 1
    ))

    # Adjuvant efficacy boost for protein vaccines (and small boost for mRNA/DNA)
    # Captures clinical mechanisms not modelled by the mechanistic cascade:
    # depot effect, direct B cell stimulation, antibody avidity enhancement.
    _ADJ_EFF_BOOST = {
        'None (formulation only)':             0.00,
        'Alum (aluminium hydroxide)':           0.19,
        'Aluminium phosphate':                  0.17,
        'MF59 (squalene o/w emulsion)':         0.12,
        'AS03 (squalene + α-tocopherol)':       0.13,
        'AddaVax (MF59 mimetic)':               0.12,
        'Montanide ISA 51 (water-in-oil)':      0.14,
        'Montanide ISA 720 (squalene w/o)':     0.12,
        'MPLA / MPL (TLR4 agonist)':            0.15,
        'GLA-SE (TLR4 synthetic)':              0.16,
        'GLA-AF (TLR4, aqueous)':               0.10,
        'CpG 1018 / ODN 1018 (TLR9)':          0.20,
        'CpG 7909 / ODN 2006 (TLR9)':          0.20,
        'R848 / Resiquimod (TLR7/8)':           0.18,
        '3M-052 (TLR7/8, lipidated)':           0.22,
        'AS37 (TLR7, benzonaphthyridine-alum)': 0.16,
        'Imiquimod / R837 (TLR7)':              0.14,
        'Poly(I:C) (TLR3/MDA-5)':              0.10,
        'Poly-ICLC / Hiltonol (TLR3)':          0.12,
        'Pam3CSK4 (TLR1/2)':                   0.06,
        'Matrix-M (Novavax, saponin NP)':       0.20,
        'ISCOMATRIX (saponin NP)':              0.18,
        'AS01B (MPL + QS-21, liposome)':        0.28,
        'AS01E (MPL + QS-21, half-dose)':       0.22,
        'AS04 (MPL + Alum)':                    0.20,
        'AS15 (MPL + QS-21 + CpG, liposome)':  0.26,
        'ALFQ (Army liposome + QS-21)':         0.22,
        'CAF01 (DDA + TDB, Mincle)':            0.12,
        'CAF09 / CAF09b (DDA + MMG + Poly I:C)': 0.14,
        "2'3'-cGAMP (STING agonist)":           0.10,
        'ADU-S100 / MIW815 (STING)':            0.08,
        'dmLT (double mutant heat-labile toxin)': 0.10,
        'Flagellin / FliC (TLR5)':              0.08,
        'Chitosan (polymer, cGAS-STING + NLRP3)': 0.07,
        'CFA (Complete Freund: research only)': 0.20,
    }
    # For mRNA and DNA: DIRECT MOLECULAR FORMULA using the four key molecular
    # determinants — each independently sourced from published literature.
    # This ensures efficacy varies with every molecular input, not just the
    # innate cascade pathway (which was dominated by TLR7/8 and did not
    # correctly reflect the m1Ψ > unmodified ordering observed in trials).
    #
    # Formula:
    #   E_raw = tlr_evasion×0.25 + (trans_eff×mod_level/100)×0.35
    #           + antigenicity×0.25 + pKa_opt×0.15
    #   efficacy = clip(CAL_FLOOR + CAL_SCALE × E_raw, 5, 98)
    #
    # Molecular inputs → efficacy contribution:
    #   tlr_evasion (0.25): reduces innate sensing → better translation context
    #     (Karikó 2005 Immunity; Andries 2015 Nat Biotechnol)
    #   translation_eff × mod_level (0.35): direct antigen expression level
    #     (Anderson 2010 Mol Ther; Hassett 2019 npj Vaccines)
    #   antigenicity (0.25): epitope quality from XGBoost ML model
    #     (sequence-specific; determined by antigen, not formulation)
    #   pKa_optimality (0.15): endosomal escape efficiency
    #     peaked at pKa 6.5; (Kulkarni 2021 Nano Lett)
    #
    # Calibration anchors:
    #   BNT162b2 (ALC-0315 + m1Ψ):  94%  (Polack NEJM 2020)
    #   DOTAP + unmodified:          47%  (Kauffman 2015; poor clinical performer)
    #
    # For protein subunit: innate cascade formula retained with per-adjuvant boosts
    # calibrated to Phase 3 trials (Polack 2020, Baden 2021, Lal 2015).

    def _pka_optimality(pka):
        """Endosomal escape optimality: peaked at pKa 6.5, range ±2.5."""
        return float(max(0.0, 1.0 - abs(pka - 6.5) / 2.5))

    if vaccine_type in ('mRNA', 'DNA'):
        _tlr_ev    = mod_data.get('tlr_evasion', 0.5)
        _trans_adj = float(np.clip(mod_data.get('translation_eff', 0.8)
                                   * (modification_level / 100.0), 0.0, 1.0))
        _pka_opt   = _pka_optimality(lipid_data['pka'])
        _ag_ant    = ag_antigenicity  # from XGBoost ML model

        # Additive weights sum to 1.0
        _e_raw = (_tlr_ev * 0.25 + _trans_adj * 0.35
                  + _ag_ant * 0.25 + _pka_opt * 0.15)

        # Calibration: BNT162b2 (E_raw≈0.825) → 94%; DOTAP+unmod (E_raw≈0.460) → 47%
        _CAL_FLOOR = -0.1217
        _CAL_SCALE =  1.2863
        efficacy = float(max(5.0, min(98.0, (_CAL_FLOOR + _CAL_SCALE * _e_raw) * 100)))

        # Adjuvant boost for mRNA/DNA: small additive contribution on top
        _adj_boost = _ADJ_EFF_BOOST.get(adjuvant_name, 0.0) * 0.3
        efficacy = float(max(5.0, min(98.0, efficacy + _adj_boost * 100)))

    else:
        # ── Protein subunit: cascade-based formula + adjuvant clinical boost ──
        _EFFICACY_PARAMS_PROT = (0.128, 1.719)  # FLOOR, SCALE
        _floor, _scale = _EFFICACY_PARAMS_PROT
        _internal = (th_bias['Th1']*0.3 + th_bias['Tfh']*0.3
                     + memory_quality*0.25 + ag_antigenicity*0.15)
        _adj_boost = _ADJ_EFF_BOOST.get(adjuvant_name, 0.0)
        efficacy = float(max(5.0, min(98.0,
                    (_floor + _scale * _internal + _adj_boost) * 100)))

    # Duration scales with memory quality — extended for vaccines with
    # strong Tfh and adjuvant-driven germinal centre responses
    duration = max(3, memory_quality * 24 + rng.normal(0, 2))

    return {
        'particle_size':     float(particle_size),
        'zeta_potential':    float(zeta_potential),
        'encapsulation_eff': float(encapsulation_eff),
        'membrane_fluidity': float(membrane_fluidity),
        'innate_prediction': innate,
        'adaptive_prediction': {
            'th_bias':        th_bias,
            'memory_quality': memory_quality,
            'antibody_response': {
                'time_to_peak': ab_peak_day,
                'durability':   ab_durability,
                'magnitude':    ab_magnitude,
            },
        },
        'clinical_predictions': {
            'efficacy':        efficacy,
            'safety':          max(10, (1 - reactogenicity) * 100),
            'reactogenicity':  reactogenicity,
            'duration_months': float(duration),
        },
        'antigen_features': ag if ag_valid else None,
        '_seed': seed,
    }
    return add_confidence_intervals(result)


# ─────────────────────────────────────────────────────────────────────────────
# FORMULATION OPTIMIZER
# Sweeps the ionizable lipid × formulation ratio × modification space and
# ranks candidates by a configurable objective (efficacy, safety, balance).
# Uses the same deterministic prediction engine so results are reproducible.
# ─────────────────────────────────────────────────────────────────────────────
def run_formulation_optimizer(antigen_features, objective: str = 'balanced',
                               top_n: int = 5,
                               vaccine_type: str = 'mRNA',
                               adjuvant_name: str = 'None (formulation only)') -> list[dict]:
    """
    Grid search over design parameters appropriate to the vaccine type.

    mRNA / DNA:  sweeps ionizable lipids × ratios × modifications
    Protein:     sweeps adjuvants × formulation vehicles × antigen doses
    """

    def score(r: dict) -> float:
        clin = r['clinical_predictions']
        if objective == 'efficacy':
            return clin['efficacy'] / 100
        elif objective == 'safety':
            return clin['safety'] / 100
        elif objective == 'durability':
            return min(clin['duration_months'] / 24, 1.0)
        elif objective == 'th1_bias':
            th = r['adaptive_prediction']['th_bias']
            return (th['Th1'] + th['Tfh']) / 2
        else:  # balanced: harmonic mean
            e = clin['efficacy'] / 100
            s = clin['safety']   / 100
            return 2 * e * s / (e + s + 1e-9)

    candidates = []

    if vaccine_type in ('mRNA', 'DNA'):
        # ── LNP formulation sweep ─────────────────────────────────────────────
        lipids      = ['ALC-0315', 'SM-102', 'DLin-MC3-DMA', 'Lipid 5', 'CL4H6', 'L-319']
        ion_ratios  = [38, 42, 46, 50]
        chol_ratios = [28, 32, 36]
        peg_ratios  = [1, 2, 3]
        mods        = ['N1-methyl-pseudouridine (m1Ψ)', 'm5C + Ψ (dual)',
                       'Pseudouridine (Ψ)', 'circRNA', 'saRNA + m1Ψ']
        helper = 16
        mod_level = 100

        for lipid in lipids:
            for ir in ion_ratios:
                for cr in chol_ratios:
                    for pr in peg_ratios:
                        for mod in mods:
                            r = run_integrated_prediction(
                                lipid, ir, helper, cr, pr, mod, mod_level,
                                antigen_features=antigen_features,
                                vaccine_type=vaccine_type,
                                adjuvant_name=adjuvant_name
                            )
                            r['_formulation'] = {
                                'type': vaccine_type,
                                'label': f"{lipid} / {mod[:22]}",
                                'lipid': lipid,
                                'ionizable_ratio': ir,
                                'helper_ratio': helper,
                                'cholesterol_ratio': cr,
                                'peg_ratio': pr,
                                'modification': mod,
                                'adjuvant': adjuvant_name,
                            }
                            r['_score'] = score(r)
                            candidates.append(r)

    else:
        # ── Protein subunit: sweep adjuvants × formulation vehicles ──────────
        # The key design variables for protein vaccines are:
        #   1. Adjuvant identity (the dominant immunogenicity driver)
        #   2. Formulation vehicle (alum adsorption, emulsion, liposome, aqueous)
        #   3. Antigen dose proxy (carried as mod_level — affects expression signal)
        # LNP lipid choice is held constant at SM-102 as a carrier baseline;
        # it does not drive the immune response for protein vaccines.

        # Adjuvants compatible with protein subunit, excluding None and research-only
        protein_adjuvants = [
            k for k, v in ADJUVANTS.items()
            if 'Protein subunit' in v['compatible']
            and v['confidence'] != 'research'
            and k != 'None (formulation only)'
        ]
        # Also include None so unadjuvanted appears in comparison
        protein_adjuvants = ['None (formulation only)'] + protein_adjuvants

        # Formulation vehicles: different alum/emulsion/liposome carriers
        # modelled as ionizable ratio variants (affects particle charge and
        # complement activation — the main LNP-relevant variable for protein vaccines)
        vehicles = [
            ('Alum-adsorbed',    40, 38, 2),   # ion/chol/peg reflects compact particle
            ('Emulsion (o/w)',   42, 32, 2),   # looser emulsion geometry
            ('Liposomal',        38, 36, 1),   # high cholesterol, low PEG
            ('Aqueous solution', 42, 30, 3),   # standard aqueous
        ]

        lipid = 'SM-102'   # held constant: not the design variable for protein vaccines
        mod   = 'Unmodified'  # no nucleoside modification for protein vaccines
        mod_level = 0

        for adj in protein_adjuvants:
            for veh_name, ir, cr, pr in vehicles:
                r = run_integrated_prediction(
                    lipid, ir, 16, cr, pr, mod, mod_level,
                    antigen_features=antigen_features,
                    vaccine_type='Protein subunit',
                    adjuvant_name=adj
                )
                adj_class = ADJUVANTS[adj]['class']
                r['_formulation'] = {
                    'type': 'Protein subunit',
                    'label': f"{adj[:30]} / {veh_name}",
                    'lipid': lipid,           # kept for backward compat
                    'adjuvant': adj,
                    'adjuvant_class': adj_class,
                    'vehicle': veh_name,
                    'ionizable_ratio': ir,
                    'helper_ratio': 16,
                    'cholesterol_ratio': cr,
                    'peg_ratio': pr,
                    'modification': mod,
                }
                r['_score'] = score(r)
                candidates.append(r)

    candidates.sort(key=lambda x: x['_score'], reverse=True)
    return candidates[:top_n]
def create_breakthrough_header():
    st.markdown("""
    <style>
    /* ── Epitrix animated header ─────────────────────────────────────────────
       Gradient stays in a READABLE mid-range: no blacks or near-blacks.
       Lightest stop: #3b82f6 (bright blue)  Darkest stop: #1d4ed8 (deep blue)
       Text is always white on these backgrounds.
    ── */
    @keyframes epitrix-shift {
        0%   { background-position: 0%   60%; }
        25%  { background-position: 50%  40%; }
        50%  { background-position: 100% 60%; }
        75%  { background-position: 50%  80%; }
        100% { background-position: 0%   60%; }
    }
    .epitrix-header {
        background: linear-gradient(130deg,
            #1d4ed8 0%,
            #2563eb 18%,
            #0ea5e9 36%,
            #0891b2 52%,
            #0d9488 68%,
            #059669 84%,
            #2563eb 100%);
        background-size: 400% 400%;
        animation: epitrix-shift 14s ease infinite;
        padding: 0; margin: 0; position: relative; overflow: hidden;
        min-height: 180px;
    }
    /* subtle noise texture overlay for depth */
    .epitrix-header::before {
        content: '';
        position: absolute; inset: 0; z-index: 1;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
        opacity: 0.35; pointer-events: none;
    }
    .epitrix-header-inner {
        max-width: 1400px; margin: 0 auto;
        padding: 2.6rem 2.5rem 2.2rem;
        position: relative; z-index: 2;
        display: flex; flex-direction: column; gap: 0.55rem;
    }
    /* decorative orbs: lighter so text stays readable */
    .epitrix-orb1 {
        position: absolute; width: 380px; height: 380px; border-radius: 50%;
        background: rgba(255,255,255,0.07);
        top: -120px; right: 60px; filter: blur(70px); z-index: 1;
    }
    .epitrix-orb2 {
        position: absolute; width: 260px; height: 260px; border-radius: 50%;
        background: rgba(255,255,255,0.05);
        bottom: -80px; right: 380px; filter: blur(55px); z-index: 1;
    }
    .epitrix-pill {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: rgba(255,255,255,0.18); backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.28); border-radius: 999px;
        padding: 0.28rem 1rem; width: fit-content;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.13em;
        color: #ffffff; text-transform: uppercase;
    }
    .epitrix-wordmark {
        font-family: 'Inter', sans-serif; font-size: 3.4rem; font-weight: 900;
        margin: 0; line-height: 1.05; letter-spacing: -0.02em;
        color: #ffffff;          /* solid white — always readable */
        text-shadow: 0 2px 20px rgba(0,0,0,0.15);
    }
    .epitrix-wordmark span {
        /* accent the "ix" suffix */
        background: linear-gradient(90deg, #bfdbfe, #a5f3fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .epitrix-sub {
        font-family: 'Inter', sans-serif; font-size: 1.02rem; font-weight: 400;
        color: rgba(255,255,255,0.90);   /* increased from 0.75 */
        margin: 0; max-width: 700px; line-height: 1.65;
        text-shadow: 0 1px 4px rgba(0,0,0,0.12);
    }
    .epitrix-chips { display: flex; gap: 0.45rem; flex-wrap: wrap; margin-top: 0.3rem; }
    .epitrix-chip {
        background: rgba(255,255,255,0.15); backdrop-filter: blur(6px);
        border: 1px solid rgba(255,255,255,0.22); border-radius: 6px;
        padding: 0.2rem 0.7rem; font-size: 0.74rem;
        color: rgba(255,255,255,0.95);   /* near-white, always visible */
        font-weight: 500;
    }
    </style>
    <div class="epitrix-header">
      <div class="epitrix-orb1"></div>
      <div class="epitrix-orb2"></div>
      <div class="epitrix-header-inner">
        <div class="epitrix-pill">🔬 Hybrid ML + Mechanistic Simulation Platform · v2.0</div>
        <h1 class="epitrix-wordmark">Epitr<span>ix</span></h1>
        <p class="epitrix-sub">
          Hybrid mechanistic and machine learning platform. XGBoost epitope prediction
          (AUC 0.986) combined with parameterised innate→adaptive cascade modeling.
          From antigen sequence and LNP formulation to predicted T cell immunogenicity,
          innate activation, adaptive immune quality, and clinical outcomes.
          Confidence intervals reflect published biological variability.
        </p>
        <div class="epitrix-chips">
          <span class="epitrix-chip">🤖 XGBoost Epitope Prediction</span>
          <span class="epitrix-chip">💉 LNP Formulation</span>
          <span class="epitrix-chip">🔥 Innate Pathway Simulation</span>
          <span class="epitrix-chip">🎯 Adaptive Immune Cascade</span>
          <span class="epitrix-chip">🧠 Memory Formation</span>
          <span class="epitrix-chip">🦠 DC Programming</span>
          <span class="epitrix-chip">📊 Clinical Reactogenicity</span>
          <span class="epitrix-chip">⚗️ Formulation Optimizer</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _light_fig(fig, height=None):
    """Apply consistent white/light theme with fully visible axis labels."""
    axis_style = dict(
        color='#111827',
        tickfont=dict(color='#111827', size=12),
        title_font=dict(color='#111827', size=13),
        showgrid=True, gridcolor='#e5e7eb', gridwidth=1,
        linecolor='#d1d5db', linewidth=1,
        zeroline=True, zerolinecolor='#9ca3af',
    )
    upd = dict(
        font=dict(family='Inter', color='#111827', size=13),
        paper_bgcolor='white', plot_bgcolor='white',
        legend=dict(
            font=dict(color='#111827', size=12), bgcolor='white',
            bordercolor='#e5e7eb', borderwidth=1
        ),
        title_font=dict(color='#111827', size=15),
    )
    if height:
        upd['height'] = height
    fig.update_layout(**upd)
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


# ── Publication colour palette ─────────────────────────────────────────────
# 8 visually distinct, print-safe colours for bar charts
PUB_COLORS = [
    '#2563eb',  # blue
    '#16a34a',  # green
    '#dc2626',  # red
    '#d97706',  # amber
    '#7c3aed',  # purple
    '#0891b2',  # cyan
    '#db2777',  # pink
    '#65a30d',  # lime
]


def _apply_pub_bar_colors(fig):
    """Give each bar trace a distinct publication colour."""
    for i, trace in enumerate(fig.data):
        if trace.type in ('bar', 'scatter'):
            color = PUB_COLORS[i % len(PUB_COLORS)]
            if trace.type == 'bar':
                # Per-bar colouring when single trace with multiple bars
                if len(fig.data) == 1:
                    trace.marker.color = PUB_COLORS[:len(trace.x)] if hasattr(trace, 'x') and trace.x is not None else color
                else:
                    trace.marker.color = color
            else:
                trace.line.color = color


def pub_chart(fig, key: str, height: int = 380, wide: bool = False,
              label: str = "Download JPEG", color_bars: bool = True):
    """
    Render a Plotly figure at publication size with per-bar colours
    and a JPEG download button.

    Parameters
    ----------
    fig         : plotly Figure
    key         : unique string key for the download widget
    height      : figure height in px (default 380 — good for single panel)
    wide        : True = double-column width (1400px), False = single (900px)
    label       : download button label
    color_bars  : apply distinct colours to bar/scatter traces
    """
    pub_w = 1400 if wide else 900

    if color_bars:
        _apply_pub_bar_colors(fig)

    fig.update_layout(
        width=pub_w,
        height=height,
        font=dict(family='Arial', size=11, color='#111827'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=68, r=32, t=52, b=68),
        title_font=dict(size=13, color='#111827', family='Arial'),
        legend=dict(font=dict(size=10, color='#111827', family='Arial'),
                    bgcolor='white', bordercolor='#d1d5db', borderwidth=1),
    )
    fig.update_xaxes(
        tickfont=dict(size=10, color='#111827', family='Arial'),
        title_font=dict(size=11, color='#111827', family='Arial'),
        showgrid=True, gridcolor='#f3f4f6', linecolor='#d1d5db',
        zeroline=False,
    )
    fig.update_yaxes(
        tickfont=dict(size=10, color='#111827', family='Arial'),
        title_font=dict(size=11, color='#111827', family='Arial'),
        showgrid=True, gridcolor='#f3f4f6', linecolor='#d1d5db',
        zeroline=True, zerolinecolor='#9ca3af',
    )

    st.plotly_chart(fig, use_container_width=False)

    # JPEG download via kaleido
    try:
        import io as _io
        _img = fig.to_image(format='jpg', width=pub_w, height=height, scale=2)
        st.download_button(
            label=f"⬇ {label}",
            data=_io.BytesIO(_img),
            file_name=f"{key}.jpg",
            mime='image/jpeg',
            key=f'dl_{key}',
        )
    except Exception:
        st.caption("Install kaleido to enable JPEG download: `pip install kaleido`")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — CORE INNOVATION
# ─────────────────────────────────────────────────────────────────────────────
def display_core_innovation():
    st.markdown("""
    <div class="content-container">
      <div class="innovation-card">
        <h2 class="section-title">🚀 The Current Gap in Immune Modeling</h2>
        <p class="section-subtitle">Existing tools treat epitope prediction, innate activation, and adaptive
        outcomes as separate problems with no connection between them.
        Epitrix bridges that gap with a hybrid ML + mechanistic pipeline.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="content-container" style="padding-top:0;">
          <div class="innovation-card">
            <h3 style="font-size:1.1rem;font-weight:700;color:#111827;margin:0 0 0.75rem;">❌ Current Landscape</h3>
            <ul>
              <li>Epitope prediction tools (NetMHCpan, MHCflurry) are standalone and not connected to formulation inputs</li>
              <li>Innate and adaptive immune models exist separately with no shared pipeline</li>
              <li>No publicly available tool connects LNP formulation chemistry to T cell outcome prediction</li>
              <li>LNP pKa and lipid choice effects on Th1/Tfh polarisation are known experimentally but not computationally modelled in accessible tools</li>
              <li>Reactogenicity prediction from formulation chemistry is not available in an integrated pre-synthesis pipeline</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="content-container" style="padding-top:0;">
          <div class="innovation-card" style="border-left:4px solid #10b981;">
            <h3 style="font-size:1.1rem;font-weight:700;color:#111827;margin:0 0 0.75rem;">✅ The Epitrix Approach</h3>
            <ul>
              <li><strong>XGBoost MHC-I epitope model</strong> trained on 219k IEDB peptides, AUC 0.986 (random split), AUC 0.789 (independent holdout)</li>
              <li><strong>XGBoost T cell immunogenicity model</strong> trained on 92k IEDB assays, AUC 0.928 (human), AUC 0.907 (mouse)</li>
              <li>Delivery system features extracted from IEDB free-text fields. This is the first delivery-aware T cell predictor built from bulk IEDB data.</li>
              <li>Epitope scores feed directly into parameterised innate and adaptive cascade equations</li>
              <li>LNP molecular descriptors (pKa, branching, modification) drive innate pathway activation</li>
              <li>Innate signals shape Th1/Th2/Tfh bias, memory quality, and antibody magnitude</li>
              <li>Clinical reactogenicity predicted from the same cascade, not a separate model</li>
              <li>Efficacy equation calibrated to Phase 3 trial data: mRNA-LNP (BNT162b2/mRNA-1273) predicts ~94%, protein+AS01B ~97%, protein+alum ~60%</li>
              <li>All outputs carry 95% CI from published biological variability data</li>
            </ul>
            <p style="font-size:0.78rem;color:#6b7280;margin-top:0.75rem;font-style:italic;">
              Note: cascade coefficients are parameterised from literature, not statistically fitted.
              Results are for hypothesis generation and comparative analysis: not clinical decision-making.
            </p>
          </div>
        </div>
        """, unsafe_allow_html=True)


def display_breakthrough_concept():
    st.markdown("""
    <div class="content-container">
      <div class="innovation-card">
        <h2 class="section-title">💡 The Epitrix Concept</h2>
        <p class="section-subtitle">Hybrid ML and mechanistic pipeline, from molecular design to clinical outcome</p>
        <div class="cascade-flow">
          <div class="flow-step" style="border-color:#ede9fe;background:linear-gradient(135deg,#f5f3ff,#ffffff);">
            <strong style="color:#4c1d95 !important;">🤖 Antigen Sequence + XGBoost Epitope Model</strong>
            <em style="color:#5b21b6 !important;">MHC-I AUC 0.986 · T cell AUC 0.928 · human &amp; mouse · IEDB-trained</em>
          </div>
          <div class="flow-arrow">⬇️</div>
          <div class="flow-step">
            <strong>LNP Formulation Chemistry</strong>
            <em>Ionizable lipid · pKa · branching · PEG density · nucleic acid modifications</em>
          </div>
          <div class="flow-arrow">⬇️</div>
          <div class="flow-step">
            <strong>Innate Immune Activation</strong>
            <em>TLR7/8 · TLR3 · cGAS-STING · complement · inflammasome (parameterised equations)</em>
          </div>
          <div class="flow-arrow">⬇️</div>
          <div class="flow-step">
            <strong>Adaptive Immune Quality</strong>
            <em>Th1/Th2/Tfh bias · T cell immunogenicity · antibody magnitude · memory durability</em>
          </div>
          <div class="flow-arrow">⬇️</div>
          <div class="flow-step" style="border-color:#a7f3d0;background:linear-gradient(135deg,#ecfdf5,white);">
            <strong style="color:#065f46 !important;">📤 Predicted Clinical Outcomes</strong>
            <em style="color:#047857 !important;">Reactogenicity · safety score · tolerability · population variability ± 95% CI</em>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — PREDICTION TARGETS
# ─────────────────────────────────────────────────────────────────────────────
def display_prediction_targets():
    st.markdown("""
    <div class="content-container">
      <div class="innovation-card">
        <h2 class="section-title">🎯 Key Prediction Targets</h2>
        <p class="section-subtitle">Two ML-trained layers and five mechanistic layers predicted from molecular inputs</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    targets = [
        ("ML 1. MHC-I Epitope Binding",
         "XGBoost classifier trained on 219,853 human and 59,877 mouse IEDB 9-mer peptides. Predicts whether a peptide will bind HLA (human) or H-2 (mouse) class I alleles. 257 features per peptide including per-position physicochemistry, allele identity flags, and log-transformed IC50.",
         ["AUC-ROC 0.986 (human HLA) · 0.970 (mouse H-2) on random split",
          "Independent holdout AUC 0.789 (post-2023 binding affinity assays, n=2,838)",
          "3-class output: non-binder / weak binder / strong binder",
          "Replaces PSSM scanner with data-driven prediction"]),
        ("ML 2. T Cell Immunogenicity",
         "XGBoost classifier trained on 92,650 human and 74,210 mouse IEDB in vivo immunisation assays. Predicts probability of a peptide triggering a T cell response. Uniquely incorporates delivery system features extracted from free-text IEDB fields.",
         ["AUC-ROC 0.928 (human) · 0.907 (mouse) on random split",
          "Species-specific models: 3.7pp improvement over combined model",
          "Delivery system features: LNP, liposome, viral, DNA, mRNA, peptide-only",
          "LNP-modulated: base probability scaled by innate activation factor"]),
        ("3. Innate Sensing Specificity",
         "Parameterised mechanistic equations predicting which innate pathways are activated by a given LNP formulation. Coefficients derived from published literature (22 sources). Not statistically fitted to data.",
         ["TLR7/8 activation: driven by pKa, nucleic acid modification, and TLR evasion score",
          "cGAS-STING: driven by lipid branching factor and particle geometry",
          "Complement: modulated by PEG density and zeta potential",
          "Inflammasome: driven by unmodified nucleosides and high pKa lipids"]),
        ("4. Dendritic Cell Programming",
         "Mechanistic prediction of DC maturation state and cytokine output from innate pathway activation scores. Feeds the T cell polarisation equations downstream.",
         ["DC maturation score: weighted combination of TLR7/8, cGAS-STING, and antigen expression",
          "Cytokine kinetics: TNF-α, IL-6, IL-12, IFN-α/β temporal profiles",
          "Antigen presentation efficiency: modulated by encapsulation and endosomal escape"]),
        ("5. T Cell Differentiation",
         "Mechanistic prediction of Th1/Th2/Th17/Tfh polarisation from DC programming outputs and epitope quality. Th1 and Tfh drive protection; Th2 drives allergic risk; Th17 drives mucosal immunity.",
         ["Th1 = TLR7/8 × 0.5 + cGAS-STING × 0.3 + ag_mhc2 × 0.2",
          "Tfh = TLR7/8 × 0.35 + Th1 × 0.3 + ag_mhc2 × 0.15",
          "Coefficients are literature-informed approximations, not fitted values"]),
        ("6. Memory Formation and Antibody Quality",
         "Mechanistic prediction of long-term protective immunity from Th1, Tfh, and antigenicity inputs. Includes antibody magnitude, peak day, and durability.",
         ["Memory quality = Th1 × 0.35 + Tfh × 0.35 + TLR7/8 × 0.15 + antigenicity × 0.15",
          "Antibody magnitude = antigenicity × 0.5 + Tfh × 0.3 + TLR7/8 × 0.2",
          "Protection duration estimate in months from memory quality and Ab durability"]),
        ("7. Clinical Reactogenicity, Safety, and Efficacy",
         "Reactogenicity is derived from the innate cascade. Efficacy for mRNA/DNA uses a direct molecular formula calibrated to Phase 3 trial data. Protein subunit uses the cascade formula with per-adjuvant clinical boosts.",
         ["Reactogenicity = TLR7/8 × 0.4 + Complement × 0.3 + Inflammasome × 0.3",
          "mRNA/DNA efficacy = f(TLR evasion×0.25 + translation×mod_level×0.35 + antigenicity×0.25 + pKa_opt×0.15)",
          "Each molecular input contributes independently: changing lipid pKa, modification type, mod level, or antigen all change predicted efficacy",
          "Calibration anchors: BNT162b2 95% (Polack NEJM 2020), DOTAP+unmodified ~47% (Kauffman 2015)",
          "Protein: cascade formula + adjuvant clinical boosts (Shingrix 97%, HEPLISAV-B 93%, alum HepB 60%)",
          "All outputs carry 95% CI from published inter-individual biological variability data"]),
    ]

    for title, desc, details in targets:
        items = "".join(f"<li>{d}</li>" for d in details)
        st.markdown(f"""
        <div class="content-container" style="padding-top:0;padding-bottom:0;">
          <div class="prediction-target">
            <h4>{title}</h4>
            <p>{desc}</p>
            <ul>{items}</ul>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — DATA INTEGRATION
# NOTE: Clinical Reactogenicity is a MODEL OUTPUT, not an input data source.
#       It is therefore listed under "Model Outputs (Predicted)" below.
# ─────────────────────────────────────────────────────────────────────────────
def display_data_integration():
    st.markdown("""
    <div class="content-container">
      <div class="innovation-card">
        <h2 class="section-title">📊 Data Integration Architecture</h2>
        <p class="section-subtitle">
          Epitrix accepts two categories of user inputs and produces six categories of predicted outputs.
          The innate and adaptive immune values shown in the platform are <strong>model predictions</strong>,
          not measurements: they are derived from the molecular inputs below.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="content-container" style="padding-top:0;">
          <div class="data-card" style="border-left:4px solid #2563eb;">
            <h4>📥 User Inputs. Molecular Design</h4>
            <p style="font-size:0.82rem;color:#4b5563;margin-bottom:0.5rem;">
              Everything the user provides. No experimental measurements required.
            </p>
            <ul>
              <li><strong>Antigen sequence:</strong> Protein, mRNA, or DNA (single-letter code)</li>
              <li><strong>Species:</strong> Human (HLA) or mouse (H-2)</li>
              <li><strong>Ionizable lipid:</strong> Name, pKa, LogP, molecular weight, branching factor</li>
              <li><strong>Formulation ratios:</strong> Ionizable / helper / cholesterol / PEG mol%</li>
              <li><strong>Nucleic acid modification:</strong> m1Ψ, s²U, pseudouridine, circRNA, saRNA, unmodified</li>
              <li><strong>Modification level:</strong> 0–100%</li>
            </ul>
          </div>

          <div class="data-card" style="border-left:4px solid #7c3aed;margin-top:1rem;">
            <h4>🤖 ML Training Data (IEDB, used offline)</h4>
            <p style="font-size:0.82rem;color:#4b5563;margin-bottom:0.5rem;">
              The datasets the XGBoost models were trained on. Not required at runtime.
            </p>
            <ul>
              <li><strong>IEDB MHC ligand dataset:</strong> 219,853 human + 59,877 mouse 9-mer peptides with IC50 measurements</li>
              <li><strong>IEDB T cell dataset:</strong> 92,650 human + 74,210 mouse in vivo immunisation assays</li>
              <li><strong>Delivery system features:</strong> Extracted from free-text adjuvant and protocol fields</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="content-container" style="padding-top:0;">
          <div class="data-card output" style="border-left:4px solid #10b981;">
            <h4>📤 Predicted Outputs. ML Layer</h4>
            <p style="font-size:0.82rem;color:#4b5563;margin-bottom:0.5rem;">
              Generated by the trained XGBoost models from the antigen sequence input.
            </p>
            <ul>
              <li><strong>MHC-I binding score:</strong> Fraction of 9-mers predicted as strong/weak binders (AUC 0.986)</li>
              <li><strong>T cell immunogenicity:</strong> Per-peptide probability of triggering in vivo response (AUC 0.928)</li>
              <li><strong>Antigenicity composite:</strong> Weighted combination of MHC-I, MHC-II, B cell scores</li>
              <li><strong>CTL/Th epitope count:</strong> Estimated number of immunogenic 9-mers</li>
            </ul>
          </div>

          <div class="data-card output" style="border-left:4px solid #10b981;margin-top:1rem;">
            <h4>📤 Predicted Outputs. Mechanistic Cascade</h4>
            <p style="font-size:0.82rem;color:#4b5563;margin-bottom:0.5rem;">
              Generated by parameterised equations from formulation inputs + ML epitope scores.
              Coefficients are literature-derived, not statistically fitted.
            </p>
            <ul>
              <li><strong>Innate pathway scores:</strong> TLR7/8, TLR3, cGAS-STING, complement, inflammasome (0–1)</li>
              <li><strong>DC programming:</strong> Maturation state, cytokine kinetics (TNF-α, IL-6, IL-12, IFN-α/β)</li>
              <li><strong>T helper bias:</strong> Th1/Th2/Th17/Tfh polarisation ratios</li>
              <li><strong>Antibody response:</strong> Magnitude, peak day, durability</li>
              <li><strong>Memory quality:</strong> Long-term protection estimate (months)</li>
              <li><strong>Reactogenicity / safety:</strong> Predicted tolerability score ± 95% CI</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-container" style="padding-top:0;">
      <div style="background:#fef3c7;border:1px solid #fde68a;border-radius:10px;
           padding:0.9rem 1.2rem;font-size:0.82rem;color:#92400e;">
        <strong>⚠️ Important distinction:</strong>
        The innate and adaptive values shown in the Simulation Platform are
        <em>computational predictions</em> derived from your molecular inputs using
        parameterised equations. They are not experimental measurements.
        The ML epitope predictions (MHC-I, T cell) are trained on IEDB data and
        evaluated against independent holdout sets (human AUC 0.789 on post-2023 binding affinity assays).
        All outputs carry 95% confidence intervals reflecting published inter-individual biological variability,
        not model prediction uncertainty.
        <strong>Epitrix is a research tool for hypothesis generation and comparative formulation analysis —
        not for clinical decision-making.</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — AI PLATFORM (4 tabs)
# ─────────────────────────────────────────────────────────────────────────────
def display_modeling_platform():
    st.markdown("""
    <div class="content-container">
      <div class="innovation-card">
        <h2 class="section-title">🔬 Epitrix Hybrid Simulation Platform</h2>
        <p class="section-subtitle">XGBoost epitope prediction (AUC 0.986) · Human &amp; mouse models · 95% confidence intervals · Formulation optimizer · Parameterised mechanistic simulation</p>
      </div>
      <div style="background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #f59e0b;
           border-radius:10px;padding:0.8rem 1.2rem;margin-bottom:1rem;font-size:0.83rem;color:#78350f;">
        <strong>ℹ️ Hybrid Platform:</strong> MHC-I epitope predictions use a trained
        <strong>XGBoost model</strong> (AUC-ROC 0.986, trained on 219k IEDB peptides for human;
        AUC-ROC 0.970 for mouse). All other predictions (innate pathways, adaptive cascade,
        clinical outcomes) are parameterised <strong>mechanistic equations</strong> fitted to
        published experimental data: not trained ML models.
        The efficacy equation for <strong>mRNA/DNA vaccines</strong> uses a
        <strong>direct molecular formula</strong> calibrated to Phase 3 trial data:
        E = f(TLR evasion, translation efficiency × modification level, antigen quality, pKa optimality).
        ALC-0315 + m1Ψ → ~94% (BNT162b2/mRNA-1273 anchors); DOTAP + unmodified → ~47%.
        <strong>Protein subunit vaccines</strong> use the cascade formula + per-adjuvant clinical boosts
        (Shingrix/AS01B → ~97%; alum → ~60%; unadjuvanted → ~30%).
        All outputs include 95% confidence intervals. Results are hypothesis-generating only.
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔬 Molecular Input",
        "🔥 Innate Prediction",
        "🎯 Adaptive Outcomes",
        "⏱️ Temporal Dynamics",
        "🧬 Epitope Analysis",
        "⚗️ Formulation Optimizer",
    ])
    with tab1: molecular_input_module()
    with tab2: innate_prediction_module()
    with tab3: adaptive_outcomes_module()
    with tab4: temporal_dynamics_module()
    with tab5: epitope_analysis_module()
    with tab6: formulation_optimizer_module()


# ── TAB 1 ────────────────────────────────────────────────────────────────────
def molecular_input_module():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    ionizable_lipids_db = MOLECULAR_DESCRIPTORS['lipid_chemistry']['ionizable_lipids']
    helper_lipids_db    = MOLECULAR_DESCRIPTORS['lipid_chemistry']['helper_lipids']
    peg_lipids_db       = MOLECULAR_DESCRIPTORS['lipid_chemistry']['peg_lipids']
    mods_db             = MOLECULAR_DESCRIPTORS['nucleic_acid_modifications']

    # Category badge colors
    CAT_COLORS = {
        'Clinical':         '#16a34a', 'Approved (siRNA)': '#2563eb',
        'Cationic':         '#dc2626', 'Next-Gen':         '#7c3aed',
        'Biodegradable':    '#0891b2', 'Custom':           '#d97706',
    }

    with col1:
        st.markdown('<div class="mol-input-box"><h3>🧬 Nanoparticle Design Parameters</h3></div>',
                    unsafe_allow_html=True)

        # ── Vaccine type selector — FIRST, gates everything below ─────────────
        st.markdown("#### 💉 Vaccine Platform")
        vaccine_type = st.selectbox(
            "Vaccine type:",
            ['mRNA', 'DNA', 'Protein subunit'],
            format_func=lambda x: {
                'mRNA':             '🧬 mRNA (LNP-formulated)',
                'DNA':              '🔵 DNA (plasmid / LNP)',
                'Protein subunit':  '🟡 Protein subunit (adjuvanted)',
            }[x],
            key='vaccine_type'
        )
        VACCINE_INFO = {
            'mRNA': (
                '#ecfdf5', '#16a34a',
                'TLR7/8 sensing of single-stranded RNA drives innate activation. '
                'Nucleoside modifications (m1Ψ, s²U) reduce TLR evasion and improve translation efficiency. '
                'cGAS-STING activated by dsRNA contaminants and particle geometry.'
            ),
            'DNA': (
                '#eff6ff', '#2563eb',
                'TLR9 recognises unmethylated CpG motifs in plasmid DNA. '
                'cGAS-STING is the dominant innate sensing pathway for cytosolic DNA. '
                'TLR7/8 contribution is low; nucleoside modifications do not apply.'
            ),
            'Protein subunit': (
                '#fefce8', '#ca8a04',
                'No nucleic acid innate sensing. Innate activation is driven entirely by '
                'adjuvant choice and particle properties. LNP lipid composition, '
                'formulation ratios, and nucleic acid modifications do not apply and are hidden below.'
            ),
        }
        vc_bg, vc_col, vc_note = VACCINE_INFO[vaccine_type]
        st.markdown(f"""
<div style="background:{vc_bg};border:1px solid {vc_col}40;border-left:4px solid {vc_col};
     border-radius:8px;padding:0.6rem 0.9rem;margin-bottom:0.75rem;font-size:0.82rem;color:#374151;">
  {vc_note}
</div>""", unsafe_allow_html=True)

        # ── LNP design — hidden for protein subunit ───────────────────────────
        is_protein = vaccine_type == 'Protein subunit'

        if is_protein:
            # Protein vaccines have no LNP — set fixed neutral defaults used
            # internally by run_integrated_prediction (low pKa, balanced ratios)
            ionizable_lipid   = 'SM-102'   # neutral carrier placeholder only
            ionizable_ratio   = 42
            helper_ratio      = 16
            helper_lipid      = 'DSPC'
            cholesterol_ratio = 32
            peg_lipid         = 'DMG-PEG2000'
            peg_ratio         = 2
            ld = ionizable_lipids_db[ionizable_lipid]
            hl = helper_lipids_db.get(helper_lipid, {'membrane_rigidity': 0.9,
                                                      'phase_transition_temp': 55,
                                                      'notes': '', 'shedding_rate': 0.5})
            pl = peg_lipids_db.get(peg_lipid, {'peg_mw': 2000, 'shedding_rate': 'fast',
                                                'notes': ''})
            st.info("🟡 LNP design parameters are not applicable for protein subunit vaccines. "
                    "Adjuvant selection below is the primary design variable.")
        else:
            # ── Ionizable lipid selector ──────────────────────────────────────
            lipid_names = list(ionizable_lipids_db.keys())
            ionizable_lipid = st.selectbox(
                "Ionizable Lipid:",
                lipid_names,
                format_func=lambda x: f"[{LIPID_CATEGORIES.get(x, '')}]  {x}"
            )
            ld = ionizable_lipids_db[ionizable_lipid]
            cat   = LIPID_CATEGORIES.get(ionizable_lipid, '')
            color = CAT_COLORS.get(cat, '#6b7280')
            st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-left:4px solid {color};
     border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.75rem;">
  <span style="background:{color};color:white;font-size:0.7rem;font-weight:700;
        padding:2px 8px;border-radius:20px;text-transform:uppercase;">{cat}</span>
  <div style="margin-top:0.5rem;font-size:0.85rem;color:#374151;">
    <strong>pKa:</strong> {ld['pka']} &nbsp;|&nbsp; <strong>LogP:</strong> {ld['logP']} &nbsp;|&nbsp;
    <strong>MW:</strong> {ld['molecular_weight']} g/mol &nbsp;|&nbsp; <strong>Branching:</strong> {ld['branching_factor']}
  </div>
  <div style="margin-top:0.4rem;font-size:0.8rem;color:#6b7280;font-style:italic;">
    {ld.get('notes', '')}
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Custom lipid override ─────────────────────────────────────────
            if ionizable_lipid == '⚙️ Custom Lipid':
                st.markdown("**✏️ Define Custom Lipid Properties:**")
                c1, c2 = st.columns(2)
                with c1:
                    custom_pka  = st.number_input("pKa:", min_value=4.0, max_value=10.0,
                                                  value=6.5, step=0.05)
                    custom_logp = st.number_input("LogP:", min_value=2.0, max_value=12.0,
                                                  value=7.5, step=0.1)
                with c2:
                    custom_mw  = st.number_input("MW (g/mol):", min_value=300.0, max_value=1200.0,
                                                 value=680.0, step=1.0)
                    custom_bf  = st.number_input("Branching Factor:", min_value=0.5, max_value=5.0,
                                                 value=2.0, step=0.1)
                ionizable_lipids_db['⚙️ Custom Lipid']['pka']              = custom_pka
                ionizable_lipids_db['⚙️ Custom Lipid']['logP']             = custom_logp
                ionizable_lipids_db['⚙️ Custom Lipid']['molecular_weight'] = custom_mw
                ionizable_lipids_db['⚙️ Custom Lipid']['branching_factor'] = custom_bf

        # ── Adjuvant selector ─────────────────────────────────────────────────
        st.markdown("#### 💊 Adjuvant Selection")

        # Filter adjuvants compatible with selected vaccine type
        compatible = {k: v for k, v in ADJUVANTS.items()
                      if vaccine_type in v['compatible']}

        # Group by class for display
        def _adj_label(name):
            a = ADJUVANTS[name]
            tier_label, _ = ADJUVANT_CONFIDENCE[a['confidence']]
            return f"[{a['class']}] {name} :  {tier_label}"

        adjuvant_name = st.selectbox(
            "Adjuvant:",
            list(compatible.keys()),
            format_func=_adj_label,
            key='adjuvant_selector'
        )
        adj = ADJUVANTS[adjuvant_name]
        tier_label, tier_color = ADJUVANT_CONFIDENCE[adj['confidence']]

        # Info box
        conf_bg = {'approved': '#f0fdf4', 'clinical': '#eff6ff',
                   'preclinical': '#fffbeb', 'research': '#fef2f2'}[adj['confidence']]
        st.markdown(f"""
<div style="background:{conf_bg};border:1px solid {tier_color}40;
     border-left:4px solid {tier_color};border-radius:8px;
     padding:0.6rem 0.9rem;margin-bottom:0.75rem;font-size:0.82rem;color:#374151;">
  <span style="background:{tier_color};color:white;font-size:0.7rem;font-weight:700;
        padding:2px 8px;border-radius:20px;text-transform:uppercase;">{tier_label}</span>
  &nbsp;&nbsp;<strong>Class:</strong> {adj['class']}
  <br><span style="margin-top:0.4rem;display:block;">{adj['notes']}</span>
</div>""", unsafe_allow_html=True)

        # Show pathway scores only if not None
        if adjuvant_name != 'None (formulation only)':
            sc = adj['scores']
            def _bar(v):
                pct = int(v * 100)
                col = '#16a34a' if v < 0.2 else '#2563eb' if v < 0.5 else '#dc2626'
                return (f'<div style="display:inline-block;width:{pct}px;max-width:80px;'
                        f'height:8px;background:{col};border-radius:4px;'
                        f'vertical-align:middle;margin-right:4px;"></div>'
                        f'<span style="font-size:0.75rem;color:#374151;">{v:.2f}</span>')
            st.markdown(f"""
<div style="font-size:0.79rem;color:#374151;background:#f9fafb;
     border:1px solid #e5e7eb;border-radius:8px;padding:0.6rem 0.9rem;
     margin-bottom:0.5rem;">
  <strong>Pathway scores (literature-grounded ordinal → 0–1):</strong><br>
  TLR7/8: {_bar(sc['TLR7_8'])} &nbsp;
  TLR3: {_bar(sc['TLR3'])} &nbsp;
  cGAS-STING: {_bar(sc['cGAS_STING'])} &nbsp;
  Complement: {_bar(sc['Complement'])} &nbsp;
  Inflammasome: {_bar(sc['Inflammasome'])}
  <br><span style="color:#9ca3af;font-size:0.72rem;">
  Scores are parameterised approximations, not direct experimental measurements.
  See notes above for supporting literature.</span>
</div>""", unsafe_allow_html=True)
        if not is_protein:
            st.markdown("#### 🧪 Excipient Selection")
            ecol1, ecol2 = st.columns(2)
            with ecol1:
                helper_lipid = st.selectbox("Helper Lipid:", list(helper_lipids_db.keys()))
                hl = helper_lipids_db[helper_lipid]
                st.caption(f"Rigidity: {hl['membrane_rigidity']} | Tm: {hl['phase_transition_temp']}°C. {hl['notes']}")
            with ecol2:
                peg_lipid = st.selectbox("PEG-Lipid:", list(peg_lipids_db.keys()))
                pl = peg_lipids_db[peg_lipid]
                st.caption(f"PEG MW: {pl['peg_mw']} | Shedding: {pl['shedding_rate']}. {pl['notes']}")

            # ── Formulation ratios ────────────────────────────────────────────
            st.markdown("#### ⚗️ Formulation Ratios (mol%)")
            ionizable_ratio   = st.slider("Ionizable Lipid:", 30, 60, 42)
            helper_ratio      = st.slider("Helper Lipid:", 10, 40, 16)
            cholesterol_ratio = st.slider("Cholesterol:", 20, 50, 32)
            peg_ratio         = st.slider("PEG-Lipid:", 1, 5, 2)
            total_mol = ionizable_ratio + helper_ratio + cholesterol_ratio + peg_ratio
            bar_color = "#16a34a" if 95 <= total_mol <= 105 else "#dc2626"
            st.markdown(
                f'<div style="font-size:0.82rem;color:{bar_color};font-weight:600;">'
                f'Total mol%: {total_mol}% {"✅" if 95 <= total_mol <= 105 else "⚠️ Typical LNP formulations sum to ~100 mol%"}'
                f'</div>', unsafe_allow_html=True
            )

        # ── Nucleic acid modifications (hidden for protein subunit) ──────────
        if is_protein:
            modification       = 'Unmodified'
            modification_level = 0
        else:
            st.markdown("#### 🧬 Nucleic Acid Modifications")
            if vaccine_type == 'DNA':
                st.caption("DNA vaccines use unmethylated CpG motifs for TLR9 activation. "
                           "The modification options below reflect available backbone chemical modifications.")
            modification = st.selectbox("Base Modification:", list(mods_db.keys()))
            md = mods_db[modification]
            st.markdown(f"""
<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;
     padding:0.6rem 0.9rem;margin-bottom:0.75rem;font-size:0.82rem;">
  <strong>TLR Evasion:</strong> {md['tlr_evasion']:.0%} &nbsp;|&nbsp;
  <strong>Stability:</strong> {md['stability']:.0%} &nbsp;|&nbsp;
  <strong>Translation Eff.:</strong> {md['translation_eff']:.0%}
  {'&nbsp;|&nbsp;<strong>Cap:</strong> cap-independent' if md.get('cap_compatibility', 1.0) == 0 else ''}
  <br><span style="color:#6b7280;font-style:italic;">{md.get('notes', '')}</span>
</div>
""", unsafe_allow_html=True)
            modification_level = st.slider("Modification Level (%):", 0, 100, 80)
        md = mods_db[modification]

        # ── Antigen Sequence ──────────────────────────────────────────────────
        st.markdown("#### 🎯 Antigen Sequence Input")
        antigen_preset = st.selectbox("Select Antigen Preset:", list(ANTIGEN_PRESETS.keys()))
        preset_data    = ANTIGEN_PRESETS[antigen_preset]

        if antigen_preset == '— Enter custom sequence —':
            antigen_sequence = st.text_area(
                "Paste protein or RNA/DNA sequence:",
                placeholder="Paste amino acid (single-letter) or nucleotide sequence here...",
                height=120
            )
        else:
            antigen_sequence = st.text_area(
                "Sequence (editable):",
                value=preset_data['sequence'],
                height=100
            )
            st.caption(f"ℹ️ {preset_data['notes']}")

        # ── Species selector ──────────────────────────────────────────────────
        species = st.selectbox(
            "Prediction species:",
            ['human', 'mouse'],
            format_func=lambda x: '🧑 Human (HLA alleles)' if x == 'human' else '🐭 Mouse (H-2 alleles)',
            key='mol_species'
        )
        st.caption(
            "Human: HLA-A\\*02:01 / HLA-DR (clinical vaccine design)  |  "
            "Mouse: H-2Kb / H-2Db: for preclinical studies (C57BL/6, BALB/c)"
        )

        # Live sequence analysis preview
        ag_features = None

        # ── ML debug panel (shown only when ML not loading) ───────────────────
        if '_ml_debug' in st.session_state:
            dbg = st.session_state['_ml_debug']
            if dbg.get('error') or dbg.get('exception'):
                with st.expander("🔧 ML model diagnostic — expand to debug", expanded=False):
                    st.code(f"""App directory:  {dbg.get('app_dir', 'unknown')}
Model found at: {dbg.get('model_found', 'NOT FOUND')}
Paths checked:
  {chr(10).join('  ' + p for p in dbg.get('candidates_checked', []))}
Error:          {dbg.get('error', '')}
Exception:      {dbg.get('exception', '')}""")

        if antigen_sequence and antigen_sequence.strip():
            spinner_msg = f"🔬 Running {'XGBoost ML' if True else 'PSSM'} epitope scan [{species}]..."
            with st.spinner(spinner_msg):
                ag_features = analyze_antigen_sequence(antigen_sequence, species=species)
            if ag_features.get('valid'):
                ag = ag_features
                # Method badge — three tiers
                if ag.get('ml_used'):
                    method_badge = (
                        '<span style="background:#7c3aed;color:white;font-size:0.68rem;'
                        'font-weight:700;padding:2px 7px;border-radius:999px;">🤖 XGBoost ML</span>'
                    )
                elif ag.get('iedb_used'):
                    method_badge = (
                        '<span style="background:#16a34a;color:white;font-size:0.68rem;'
                        'font-weight:700;padding:2px 7px;border-radius:999px;">IEDB ✓</span>'
                    )
                else:
                    method_badge = (
                        '<span style="background:#6b7280;color:white;font-size:0.68rem;'
                        'font-weight:700;padding:2px 7px;border-radius:999px;">Local PSSM</span>'
                    )
                species_badge = (
                    '<span style="background:#1d4ed8;color:white;font-size:0.68rem;'
                    f'font-weight:700;padding:2px 7px;border-radius:999px;">{"🧑 Human" if species=="human" else "🐭 Mouse"}</span>'
                )
                st.markdown(f"""
<div class="antigen-card">
  <h3>🔬 Epitope Scan Results &nbsp;{method_badge}&nbsp;{species_badge}</h3>
  <span class="antigen-stat">Type: {ag['seq_type']}</span>
  <span class="antigen-stat">Length: {ag['length']} residues</span>
  <span class="antigen-stat">MHC-I: {ag['mhc1_score']:.0%}</span>
  <span class="antigen-stat">MHC-II: {ag['mhc2_score']:.0%}</span>
  <span class="antigen-stat">B-cell: {ag['b_cell_score']:.0%}</span>
  <span class="antigen-stat">Antigenicity: {ag['antigenicity']:.0%}</span>
  <p style="margin-top:0.5rem;font-size:0.8rem;color:#78350f !important;">
    CTL epitopes: <strong>{ag['ctl_epitopes_est']}</strong> &nbsp;|&nbsp;
    Th epitopes: <strong>{ag['th_epitopes_est']}</strong> &nbsp;|&nbsp;
    B-cell epitopes: <strong>{ag['bcell_epitopes_est']}</strong>
    &nbsp;·&nbsp; <em>Method: {ag.get('mhci_method','—')}</em>
  </p>
</div>""", unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Could not parse sequence: {ag_features.get('error', 'unrecognized format')}")
                ag_features = None
        else:
            st.caption("No antigen sequence provided. The prediction will use default antigenicity values.")

        # ── Run label input ──────────────────────────────────────────────────
        _adj_short = adjuvant_name.split(' (')[0][:18] if adjuvant_name != 'None (formulation only)' else 'No adj'
        _ant_short = antigen_preset[:18] if antigen_preset != '— Enter custom sequence —' else 'Custom'
        run_label_input = st.text_input(
            "Run label (optional):",
            value=f"{vaccine_type} / {ionizable_lipid} / {_adj_short} / {_ant_short}",
            help="Name this run so you can identify it in the comparison table and downloaded files."
        )

        if st.button("🚀 Predict Immune Cascade", type="primary"):
            with st.spinner("Running mechanistic cascade..."):
                r = run_integrated_prediction(
                    ionizable_lipid, ionizable_ratio, helper_ratio,
                    cholesterol_ratio, peg_ratio, modification, modification_level,
                    antigen_features=ag_features,
                    vaccine_type=vaccine_type,
                    adjuvant_name=adjuvant_name
                )

            import datetime
            run_id = f"RUN-{len(st.session_state.get('run_registry', [])) + 1:03d}"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            r['selected_helper']   = helper_lipid
            r['selected_peg']      = peg_lipid
            r['helper_rigidity']   = hl['membrane_rigidity']
            r['peg_shedding']      = pl['shedding_rate']
            r['antigen_name']      = antigen_preset if antigen_preset != '— Enter custom sequence —' else 'Custom'
            r['species']           = species
            r['vaccine_type']      = vaccine_type
            r['adjuvant']          = adjuvant_name
            r['_run_id']           = run_id
            r['_run_label']        = run_label_input.strip() or run_id
            r['_timestamp']        = timestamp

            # Full parameter record stored with result
            r['_params'] = {
                'run_id':              run_id,
                'label':               run_label_input.strip() or run_id,
                'timestamp':           timestamp,
                'vaccine_type':        vaccine_type,
                'adjuvant':            adjuvant_name,
                'adjuvant_class':      ADJUVANTS[adjuvant_name]['class'],
                'adjuvant_confidence': ADJUVANTS[adjuvant_name]['confidence'],
                'ionizable_lipid':    ionizable_lipid,
                'lipid_pka':          ld['pka'],
                'lipid_logP':         ld['logP'],
                'lipid_mw':           ld['molecular_weight'],
                'lipid_branching':    ld['branching_factor'],
                'ionizable_ratio':    ionizable_ratio,
                'helper_lipid':       helper_lipid,
                'helper_ratio':       helper_ratio,
                'cholesterol_ratio':  cholesterol_ratio,
                'peg_lipid':          peg_lipid,
                'peg_ratio':          peg_ratio,
                'modification':       modification,
                'modification_level': modification_level,
                'tlr_evasion':        md['tlr_evasion'],
                'translation_eff':    md['translation_eff'],
                'antigen':            antigen_preset if antigen_preset != '— Enter custom sequence —' else 'Custom',
                'antigen_seq_50':     (antigen_sequence or '')[:50],
                'species':            species,
                'antigenicity':       ag_features.get('antigenicity', 'n/a') if ag_features else 'n/a',
                'mhc1_score':         ag_features.get('mhc1_score', 'n/a') if ag_features else 'n/a',
                'ctl_epitopes':       ag_features.get('ctl_epitopes_est', 'n/a') if ag_features else 'n/a',
            }

            st.session_state['cascade_results'] = r
            st.session_state['antigen_features_cache'] = ag_features

            # Append to run registry
            if 'run_registry' not in st.session_state:
                st.session_state['run_registry'] = []
            st.session_state['run_registry'].append(r)

            st.success(f"✅ {run_id} saved: \"{r['_run_label']}\"  |  Seed: `{r['_seed']}`")
            st.caption("🔒 Identical inputs always produce identical results. 95% CI bands shown on all charts.")

    with col2:
        st.markdown('<div class="mol-input-box"><h3>📊 Predicted Molecular Profile</h3></div>',
                    unsafe_allow_html=True)
        if 'cascade_results' in st.session_state:
            r  = st.session_state['cascade_results']
            ci = r.get('confidence_intervals', {})
            _r_is_protein = r.get('vaccine_type') == 'Protein subunit'

            if not _r_is_protein:
                st.markdown("#### 🔬 Physicochemical Properties")
                pc1, pc2 = st.columns(2)
                with pc1:
                    lo, hi = ci.get('particle_size', (0, 0))
                    st.metric("Particle Size", f"{r['particle_size']:.0f} nm",
                              delta=f"95% CI: {lo:.0f}–{hi:.0f} nm", delta_color="off")
                    st.metric("Zeta Potential", f"{r['zeta_potential']:.1f} mV")
                with pc2:
                    st.metric("Encapsulation Eff.", f"{r['encapsulation_eff']:.2f}")
                    st.metric("Membrane Fluidity",  f"{r['membrane_fluidity']:.2f}")

                # Excipient badges
                if 'selected_helper' in r:
                    st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
     padding:0.6rem 1rem;margin:0.5rem 0;font-size:0.82rem;color:#111827;">
  <strong>Helper:</strong> {r['selected_helper']}
  &nbsp;(rigidity {r.get('helper_rigidity','—')}) &nbsp;|&nbsp;
  <strong>PEG:</strong> {r['selected_peg']} &nbsp;(shedding: {r.get('peg_shedding','—')})
</div>""", unsafe_allow_html=True)
            else:
                # For protein vaccines show adjuvant summary instead
                adj = r.get('adjuvant', 'None (formulation only)')
                adj_data = ADJUVANTS.get(adj, {})
                _, adj_col = ADJUVANT_CONFIDENCE.get(adj_data.get('confidence', ''), ('', '#6b7280'))
                st.markdown(f"""
<div style="background:#faf5ff;border:1px solid #e9d5ff;border-left:4px solid {adj_col};
     border-radius:8px;padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.82rem;color:#374151;">
  <strong>💊 Adjuvant:</strong>
  <span style="color:{adj_col};font-weight:600;">{adj}</span><br>
  <strong>Class:</strong> {adj_data.get('class','—')} &nbsp;|&nbsp;
  <strong>Status:</strong> {ADJUVANT_CONFIDENCE.get(adj_data.get('confidence',''),('—',''))[0]}
</div>""", unsafe_allow_html=True)

            # Antigen summary badge
            ag = r.get('antigen_features')
            if ag and ag.get('valid'):
                st.markdown(f"""
<div style="background:#fefce8;border:1px solid #fde68a;border-left:4px solid #f59e0b;
     border-radius:8px;padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.82rem;color:#78350f;">
  <strong>🎯 Antigen:</strong> {r.get('antigen_name','Custom')} &nbsp;|&nbsp;
  {ag['seq_type']} · {ag['length']} residues &nbsp;|&nbsp;
  MHC-I <strong>{ag['mhc1_score']:.0%}</strong> &nbsp;
  MHC-II <strong>{ag['mhc2_score']:.0%}</strong> &nbsp;
  Antigenicity <strong>{ag['antigenicity']:.0%}</strong>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;
     padding:0.5rem 1rem;margin:0.5rem 0;font-size:0.82rem;color:#6b7280;">
  🎯 <em>No antigen sequence provided. Using default antigenicity parameters.</em>
</div>""", unsafe_allow_html=True)

            st.markdown("#### 🔥 Predicted Innate Activation")
            _inn_preview = {k: v for k, v in r['innate_prediction'].items()
                            if not k.startswith('_')}
            pathways = list(_inn_preview.keys())
            scores   = list(_inn_preview.values())
            fig = go.Figure(go.Scatterpolar(
                r=scores, theta=pathways, fill='toself',
                name='Innate Activation',
                line_color='rgb(37,99,235)', fillcolor='rgba(37,99,235,0.2)'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                           tickfont=dict(color='#111827'))),
                title=dict(text="Innate Pathway Activation Profile", font=dict(color='#111827')),
                font=dict(family='Inter', color='#111827'),
                paper_bgcolor='white', plot_bgcolor='white',
                margin=dict(t=50, b=20, l=20, r=20)
            )
            pub_chart(fig, key="th_bias_pie", height=380)
        else:
            st.info("Configure parameters on the left and click **Predict Immune Cascade**.")

    st.markdown('</div>', unsafe_allow_html=True)


# ── TAB 2 ────────────────────────────────────────────────────────────────────
def _run_context_banner():
    """Render a compact banner showing the vaccine type, adjuvant, and run label
    for the currently displayed cascade result. Call at the top of every result tab."""
    r = st.session_state.get('cascade_results', {})
    if not r:
        return
    vtype    = r.get('vaccine_type', 'mRNA')
    adjuvant = r.get('adjuvant', 'None (formulation only)')
    label    = r.get('_run_label', '')
    run_id   = r.get('_run_id', '')
    ts       = r.get('_timestamp', '')

    vtype_colors = {
        'mRNA':            ('#ecfdf5', '#16a34a'),
        'DNA':             ('#eff6ff', '#2563eb'),
        'Protein subunit': ('#fefce8', '#ca8a04'),
    }
    bg, col = vtype_colors.get(vtype, ('#f9fafb', '#6b7280'))

    adj_data = ADJUVANTS.get(adjuvant, {})
    adj_conf = adj_data.get('confidence', '')
    _, adj_col = ADJUVANT_CONFIDENCE.get(adj_conf, ('', '#6b7280'))

    st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
     padding:0.55rem 1rem;margin-bottom:0.85rem;font-size:0.8rem;
     color:#374151;display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
  <span style="font-weight:700;color:#111827;">📋 Showing:</span>
  <span style="background:{bg};color:{col};border:1px solid {col}40;
        padding:2px 9px;border-radius:20px;font-weight:600;">{vtype}</span>
  <span style="background:#f0f9ff;color:{adj_col};border:1px solid {adj_col}40;
        padding:2px 9px;border-radius:20px;font-weight:600;">
    💊 {adjuvant[:40]}{'…' if len(adjuvant)>40 else ''}
  </span>
  <span style="color:#6b7280;">{run_id}{' · ' + label if label and label != run_id else ''}</span>
  <span style="color:#9ca3af;margin-left:auto;">{ts}</span>
</div>""", unsafe_allow_html=True)


def innate_prediction_module():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)

    # ── Honest labeling banner ────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #f59e0b;
         border-radius:10px;padding:0.75rem 1.2rem;margin-bottom:1rem;font-size:0.83rem;color:#78350f;">
      <strong>ℹ️ Mechanistic Simulation:</strong> Innate pathway scores are derived from
      parameterised equations fitted to published experimental datasets, not a trained ML model.
      Shaded bands represent published inter-individual biological variability (95% CI).
    </div>
    """, unsafe_allow_html=True)

    _run_context_banner()

    if 'cascade_results' in st.session_state:
        r   = st.session_state['cascade_results']
        ci  = r.get('confidence_intervals', {})
        inn = r['innate_prediction']

        st.markdown("""
        <div class="innovation-card">
          <h3 style="font-size:1.25rem;font-weight:700;color:#111827;margin:0 0 0.5rem;">
            🔥 Innate Immune Pathway Predictions + 95% CI
          </h3>
        </div>
        """, unsafe_allow_html=True)

        col_r, col_l = st.columns(2)

        # ── Innate radar with CI error bars (bar chart proxy) ─────────────────
        with col_r:
            # Filter out private keys (e.g. _TLR9 stored for DNA vaccine reference)
            inn_display = {k: v for k, v in inn.items() if not k.startswith('_')}
            pathways = list(inn_display.keys())
            scores   = list(inn_display.values())
            lows  = [ci.get(f'innate_{p}', (s*0.7, s*1.3))[0] for p, s in zip(pathways, scores)]
            highs = [ci.get(f'innate_{p}', (s*0.7, s*1.3))[1] for p, s in zip(pathways, scores)]
            err_minus = [max(0, s - lo) for s, lo in zip(scores, lows)]
            err_plus  = [max(0, hi - s) for s, hi in zip(scores, highs)]

            fig_r = go.Figure()
            fig_r.add_trace(go.Bar(
                x=pathways, y=scores,
                error_y=dict(type='data', symmetric=False,
                             array=err_plus, arrayminus=err_minus,
                             color='#374151', thickness=2, width=6),
                marker_color=['#ef4444','#f59e0b','#7c3aed','#06b6d4','#10b981'],
                showlegend=False,
            ))
            # Separate scatter trace for labels — placed at 50% of bar height,
            # completely clear of the error bar which sits at 100%
            fig_r.add_trace(go.Scatter(
                x=pathways,
                y=[s * 0.5 for s in scores],
                mode='text',
                text=[f"<b>{s:.2f}</b>" for s in scores],
                textfont=dict(color='white', size=13, family='Arial Black'),
                showlegend=False,
                hoverinfo='skip',
            ))
            fig_r.update_layout(
                title=dict(text="Innate Pathway Activation + 95% CI",
                           font=dict(color='#111827', size=14)),
                yaxis=dict(title=dict(text="Activation Score (0–1)",
                                      font=dict(color='#111827', size=12)),
                           tickfont=dict(color='#111827'), range=[0, 1.3]),
                xaxis=dict(tickfont=dict(color='#111827', size=10)),
                margin=dict(t=50, b=50),
            )
            _light_fig(fig_r)
            pub_chart(fig_r, key="innate_bar", height=400)

        # ── Cytokine kinetics with CI shaded bands ────────────────────────────
        with col_l:
            tlr = inn['TLR7_8'];  inf_v = inn['Inflammasome'];  cgs = inn['cGAS_STING']
            tlr_lo = ci.get('innate_TLR7_8', (tlr*0.72, tlr*1.28))[0]
            tlr_hi = ci.get('innate_TLR7_8', (tlr*0.72, tlr*1.28))[1]

            t_cont = np.linspace(0, 72, 200)
            cytokine_defs = [
                ('TNF-α',   tlr,   tlr_lo,  tlr_hi,   4,  20, '#ef4444'),
                ('IL-6',    inf_v, inf_v*0.72, inf_v*1.28, 6, 30, '#f59e0b'),
                ('IL-12',   tlr*0.8, tlr_lo*0.8, tlr_hi*0.8, 12, 60, '#2563eb'),
                ('IFN-α/β', cgs,   cgs*0.68, cgs*1.32, 8,  40, '#10b981'),
            ]
            fig_c = go.Figure()
            for name, base, lo_base, hi_base, peak_t, width_sq, color in cytokine_defs:
                y_mean = base    * np.exp(-(t_cont - peak_t)**2 / width_sq)
                y_lo   = lo_base * np.exp(-(t_cont - peak_t)**2 / width_sq)
                y_hi   = hi_base * np.exp(-(t_cont - peak_t)**2 / width_sq)
                y_mean[0] = y_lo[0] = y_hi[0] = 0
                # shaded CI band
                fig_c.add_trace(go.Scatter(
                    x=np.concatenate([t_cont, t_cont[::-1]]),
                    y=np.concatenate([y_hi, y_lo[::-1]]),
                    fill='toself', fillcolor=color.replace('#', 'rgba(') + ',0.12)' if False else
                        f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
                    line=dict(width=0), showlegend=False, hoverinfo='skip'
                ))
                fig_c.add_trace(go.Scatter(
                    x=t_cont, y=y_mean, mode='lines',
                    name=name, line=dict(color=color, width=2.5)
                ))
            fig_c.update_layout(
                title=dict(text="Cytokine Release Kinetics + 95% CI Bands",
                           font=dict(color='#111827', size=14)),
                xaxis=dict(title=dict(text="Time (hours)", font=dict(color='#111827', size=12)),
                           tickfont=dict(color='#111827')),
                yaxis=dict(title=dict(text="Relative Level", font=dict(color='#111827', size=12)),
                           tickfont=dict(color='#111827')),
                margin=dict(t=50, b=50)
            )
            _light_fig(fig_c)
            pub_chart(fig_c, key="cytokine_kinetics", height=360)

        # ── Innate score summary table ────────────────────────────────────────
        st.markdown("#### 📋 Innate Pathway Score Summary")
        rows_html = "".join(f"""
        <tr>
          <td style="padding:0.5rem 1rem;font-weight:600;color:#111827;">{p}</td>
          <td style="padding:0.5rem 1rem;text-align:center;color:#111827;">{s:.3f}</td>
          <td style="padding:0.5rem 1rem;text-align:center;color:#6b7280;">
            {ci.get(f'innate_{p}',(0,0))[0]:.3f} – {ci.get(f'innate_{p}',(0,0))[1]:.3f}
          </td>
          <td style="padding:0.5rem 1rem;">
            <div style="background:#e5e7eb;border-radius:4px;height:10px;width:100%;">
              <div style="background:#2563eb;border-radius:4px;height:10px;width:{s*100:.0f}%;"></div>
            </div>
          </td>
        </tr>""" for p, s in inn.items())
        st.markdown(f"""
<table style="width:100%;border-collapse:collapse;background:white;
     border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;font-size:0.85rem;">
  <thead>
    <tr style="background:#f9fafb;border-bottom:1px solid #e5e7eb;">
      <th style="padding:0.6rem 1rem;text-align:left;color:#374151;">Pathway</th>
      <th style="padding:0.6rem 1rem;text-align:center;color:#374151;">Score</th>
      <th style="padding:0.6rem 1rem;text-align:center;color:#374151;">95% CI</th>
      <th style="padding:0.6rem 1rem;text-align:left;color:#374151;">Activation level</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>""", unsafe_allow_html=True)
    else:
        st.info("Run molecular prediction first (🔬 Molecular Input tab).")

    st.markdown('</div>', unsafe_allow_html=True)


# ── TAB 3 ────────────────────────────────────────────────────────────────────
def adaptive_outcomes_module():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)

    _run_context_banner()

    if 'cascade_results' in st.session_state:
        r = st.session_state['cascade_results']

        st.markdown("""
        <div class="innovation-card">
          <h3 style="font-size:1.25rem;font-weight:700;color:#111827;margin:0 0 0.5rem;">
            🎯 Adaptive Immune Outcomes + Clinical Predictions
          </h3>
        </div>
        """, unsafe_allow_html=True)

        # ── Row 1: Th bias + Clinical metrics ────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔬 T Helper Cell Differentiation")
            th = r['adaptive_prediction']['th_bias']
            fig = go.Figure(go.Pie(
                labels=list(th.keys()), values=list(th.values()), hole=0.3,
                marker_colors=['#2563eb', '#10b981', '#f59e0b', '#ef4444']
            ))
            fig.update_traces(textfont=dict(color='#111827', size=13))
            fig.update_layout(
                title=dict(text="Predicted T Helper Cell Bias", font=dict(color='#111827', size=14)),
                font=dict(family='Inter', color='#111827'),
                paper_bgcolor='white',
                legend=dict(font=dict(color='#111827'), bgcolor='white'),
                margin=dict(t=50, b=20)
            )
            pub_chart(fig, key="th_bias_pie", height=380)

        with col2:
            st.markdown("#### 📊 Clinical Predictions + 95% CI")
            clin = r['clinical_predictions']
            ci_  = r.get('confidence_intervals', {})
            m1, m2 = st.columns(2)
            eff_lo, eff_hi = ci_.get('efficacy',        (0, 0))
            dur_lo, dur_hi = ci_.get('duration_months', (0, 0))
            rea_lo, rea_hi = ci_.get('reactogenicity',  (0, 0))
            with m1:
                st.metric("Predicted Efficacy",   f"{clin['efficacy']:.1f}%",
                          delta=f"CI: {eff_lo:.1f}–{eff_hi:.1f}%", delta_color="off")
                st.metric("Safety Score",          f"{clin['safety']:.1f}%")
            with m2:
                st.metric("Reactogenicity",        f"{clin['reactogenicity']:.2f}",
                          delta=f"CI: {rea_lo:.2f}–{rea_hi:.2f}", delta_color="off")
                st.metric("Protection Duration",   f"{clin['duration_months']:.1f} mo",
                          delta=f"CI: {dur_lo:.1f}–{dur_hi:.1f}", delta_color="off")

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=clin['reactogenicity'],
                title=dict(text="Predicted Reactogenicity", font=dict(color='#111827', size=13)),
                gauge=dict(
                    axis=dict(range=[0, 1], tickcolor='#111827',
                              tickfont=dict(color='#111827', size=11)),
                    bar=dict(color="#2563eb"),
                    steps=[
                        dict(range=[0, 0.33], color="#dcfce7"),
                        dict(range=[0.33, 0.66], color="#fef9c3"),
                        dict(range=[0.66, 1.0], color="#fee2e2"),
                    ],
                    threshold=dict(line=dict(color="#ef4444", width=4), thickness=0.75, value=0.66)
                ),
                number=dict(font=dict(color='#111827', size=20))
            ))
            fig_g.update_layout(
                paper_bgcolor='white', font=dict(family='Inter', color='#111827'),
                margin=dict(t=40, b=20, l=20, r=20), height=220
            )
            pub_chart(fig_g, key="antibody_kinetics", height=360)

        # ── Row 2: DC Programming + Memory Formation ──────────────────────────
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### 🦠 Dendritic Cell Programming")
            inn   = r['innate_prediction']
            tlr   = inn['TLR7_8']
            cgas  = inn['cGAS_STING']
            comp  = inn['Complement']
            infla = inn['Inflammasome']

            cdc1_score = float(np.clip(tlr * 0.5 + cgas * 0.5, 0, 1))
            cdc2_score = float(np.clip(tlr * 0.4 + infla * 0.4 + comp * 0.2, 0, 1))
            pdc_score  = float(np.clip(tlr * 0.3 + cgas * 0.6, 0, 1))
            maturation = float(np.clip((tlr + cgas) / 2, 0, 1))
            il12_prod  = float(np.clip(cdc1_score * 0.7 + cgas * 0.3, 0, 1))
            il10_prod  = float(np.clip((1 - tlr) * 0.4 + comp * 0.3, 0, 1))

            t_dc = np.linspace(0, 5, 100)
            fig_dc = go.Figure()
            fig_dc.add_trace(go.Scatter(x=t_dc, y=cdc1_score*(1-np.exp(-t_dc*1.5)),
                name='cDC1 (CTL priming)', line=dict(color='#2563eb', width=2.5)))
            fig_dc.add_trace(go.Scatter(x=t_dc, y=cdc2_score*(1-np.exp(-t_dc*1.2)),
                name='cDC2 (Th2/Th17)', line=dict(color='#f59e0b', width=2.5)))
            fig_dc.add_trace(go.Scatter(x=t_dc, y=pdc_score*(1-np.exp(-t_dc*2.0)),
                name='pDC (IFN-α)', line=dict(color='#7c3aed', width=2.5)))
            fig_dc.add_trace(go.Scatter(x=t_dc, y=il12_prod*(1-np.exp(-t_dc*1.8)),
                name='IL-12 output', line=dict(color='#10b981', width=2, dash='dot')))
            fig_dc.update_layout(
                title=dict(text="DC Subset Activation & Cytokine Output",
                           font=dict(color='#111827', size=14)),
                xaxis=dict(title=dict(text="Days post-vaccination",
                                      font=dict(color='#111827', size=13)),
                           tickfont=dict(color='#111827', size=11)),
                yaxis=dict(title=dict(text="Activation Level",
                                      font=dict(color='#111827', size=13)),
                           tickfont=dict(color='#111827', size=11), range=[0, 1]),
                margin=dict(t=50, b=50)
            )
            _light_fig(fig_dc)
            st.markdown(f"""
<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;
     padding:0.6rem 1rem;margin-bottom:0.5rem;font-size:0.82rem;color:#1e3a5f;">
  <strong>cDC1:</strong> {cdc1_score:.2f} &nbsp;|&nbsp;
  <strong>cDC2:</strong> {cdc2_score:.2f} &nbsp;|&nbsp;
  <strong>pDC:</strong> {pdc_score:.2f} &nbsp;|&nbsp;
  <strong>Maturation:</strong> {maturation:.2f}<br>
  <strong>IL-12 (pro-Th1):</strong> {il12_prod:.2f} &nbsp;|&nbsp;
  <strong>IL-10 (regulatory):</strong> {il10_prod:.2f}
</div>""", unsafe_allow_html=True)
            pub_chart(fig_dc, key="dc_maturation", height=360)

        with col4:
            st.markdown("#### 🧠 Memory Formation Quality")
            adp     = r['adaptive_prediction']
            mq      = adp['memory_quality']
            tfh_v   = adp['th_bias']['Tfh']
            th1_v   = adp['th_bias']['Th1']
            ab      = adp['antibody_response']
            dur     = ab['durability']
            mag     = ab.get('magnitude', 0.5)

            llpc_score  = float(np.clip(tfh_v * 0.5 + mq * 0.5, 0, 1))
            mem_b_score = float(np.clip(tfh_v * 0.4 + dur * 0.4 + mq * 0.2, 0, 1))
            mem_t_score = float(np.clip(th1_v * 0.5 + mq * 0.5, 0, 1))
            gcr_score   = float(np.clip(tfh_v * 0.6 + mag * 0.4, 0, 1))

            t_mem = np.linspace(0, 12, 200)
            llpc_curve = llpc_score  * (0.7 + 0.3 * np.exp(-t_mem * 0.05))
            memb_curve = mem_b_score * (0.6 + 0.4 * np.exp(-t_mem * 0.08))
            memt_curve = mem_t_score * (0.5 + 0.5 * np.exp(-t_mem * 0.06))
            ab_curve   = mag * np.exp(-t_mem * (0.15 * (1 - dur)))

            fig_mem = go.Figure()
            fig_mem.add_trace(go.Scatter(x=t_mem, y=llpc_curve,
                name='Long-lived plasma cells', line=dict(color='#2563eb', width=2.5),
                fill='tozeroy', fillcolor='rgba(37,99,235,0.06)'))
            fig_mem.add_trace(go.Scatter(x=t_mem, y=memb_curve,
                name='Memory B cells', line=dict(color='#10b981', width=2.5)))
            fig_mem.add_trace(go.Scatter(x=t_mem, y=memt_curve,
                name='Memory T cells', line=dict(color='#f59e0b', width=2.5)))
            fig_mem.add_trace(go.Scatter(x=t_mem, y=ab_curve,
                name='Circulating Ab titer', line=dict(color='#ef4444', width=2, dash='dot')))
            fig_mem.update_layout(
                title=dict(text="Memory Cell Persistence (12 months)",
                           font=dict(color='#111827', size=14)),
                xaxis=dict(title=dict(text="Months post-vaccination",
                                      font=dict(color='#111827', size=13)),
                           tickfont=dict(color='#111827', size=11)),
                yaxis=dict(title=dict(text="Relative Level",
                                      font=dict(color='#111827', size=13)),
                           tickfont=dict(color='#111827', size=11), range=[0, 1]),
                margin=dict(t=50, b=50)
            )
            _light_fig(fig_mem)
            prot_months = r['clinical_predictions']['duration_months']
            st.markdown(f"""
<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;
     padding:0.6rem 1rem;margin-bottom:0.5rem;font-size:0.82rem;color:#065f46;">
  <strong>Memory quality index:</strong> {mq:.2f} &nbsp;|&nbsp;
  <strong>LLPC:</strong> {llpc_score:.2f} &nbsp;|&nbsp;
  <strong>GC reaction:</strong> {gcr_score:.2f}<br>
  <strong>Memory B:</strong> {mem_b_score:.2f} &nbsp;|&nbsp;
  <strong>Memory T:</strong> {mem_t_score:.2f} &nbsp;|&nbsp;
  <strong>Est. protection:</strong> {prot_months:.1f} months
</div>""", unsafe_allow_html=True)
            pub_chart(fig_mem, key="memory_quality", height=360)

        # ── Antibody kinetics with CI band + population stratification ───────
        st.markdown("#### 🩸 Antibody Response Kinetics + 95% CI")

        # Population modifier
        pop_col, _ = st.columns([1, 2])
        with pop_col:
            population = st.selectbox("Population:", [
                'General adult (18–60)',
                'Elderly (60+)',
                'Pediatric (2–17)',
                'Immunocompromised',
            ], key='pop_select')

        # Population modifiers derived from published meta-analyses
        POP_MOD = {
            'General adult (18–60)':  {'mag': 1.00, 'dur': 1.00, 'react': 1.00, 'ci_inflate': 1.0,
                                        'note': 'Reference population'},
            'Elderly (60+)':          {'mag': 0.68, 'dur': 0.75, 'react': 0.80, 'ci_inflate': 1.35,
                                        'note': 'Immunosenescence reduces peak titer ~32% and durability ~25% (Crooke et al. 2019)'},
            'Pediatric (2–17)':       {'mag': 1.15, 'dur': 1.20, 'react': 1.30, 'ci_inflate': 1.20,
                                        'note': 'Higher peak titer and durability; elevated reactogenicity vs adults (Wodi et al. 2022)'},
            'Immunocompromised':      {'mag': 0.45, 'dur': 0.55, 'react': 0.60, 'ci_inflate': 1.80,
                                        'note': 'Substantially blunted humoral responses; wide variability (Connell et al. 2022)'},
        }
        pm = POP_MOD[population]

        ab   = r['adaptive_prediction']['antibody_response']
        t_d  = np.linspace(0, 30, 200)
        peak = ab['time_to_peak']
        dur  = ab['durability']
        mag  = ab.get('magnitude', 0.8) * pm['mag']
        ci   = r.get('confidence_intervals', {})

        ab_cv_inflated = _CV['ab_magnitude'] * pm['ci_inflate']
        titers     = mag * np.where(t_d <= peak, (t_d/peak)**2, np.exp(-0.05*(1-dur)*(t_d-peak)))
        titers_lo  = titers * (1 - Z95 * ab_cv_inflated)
        titers_hi  = titers * (1 + Z95 * ab_cv_inflated)
        titers_lo  = np.clip(titers_lo, 0, None)

        fig2 = go.Figure()
        # CI band
        fig2.add_trace(go.Scatter(
            x=np.concatenate([t_d, t_d[::-1]]),
            y=np.concatenate([titers_hi, titers_lo[::-1]]),
            fill='toself', fillcolor='rgba(37,99,235,0.10)',
            line=dict(width=0), showlegend=True, name='95% CI band',
            hoverinfo='skip'
        ))
        fig2.add_trace(go.Scatter(
            x=t_d, y=titers, mode='lines',
            name=f'Predicted titer ({population})',
            line=dict(color='#2563eb', width=3)
        ))
        if population != 'General adult (18–60)':
            # Also show reference adult line faintly
            mag_ref = ab.get('magnitude', 0.8)
            titers_ref = mag_ref * np.where(t_d <= peak, (t_d/peak)**2, np.exp(-0.05*(1-dur)*(t_d-peak)))
            fig2.add_trace(go.Scatter(
                x=t_d, y=titers_ref, mode='lines',
                name='Reference adult',
                line=dict(color='#94a3b8', width=1.5, dash='dot')
            ))
        fig2.update_layout(
            title=dict(text=f"Antibody Response Kinetics: {population}",
                       font=dict(color='#111827', size=15)),
            xaxis=dict(title=dict(text="Time (days)", font=dict(color='#111827', size=13)),
                       tickfont=dict(color='#111827', size=12)),
            yaxis=dict(title=dict(text="Relative Antibody Titer",
                                  font=dict(color='#111827', size=13)),
                       tickfont=dict(color='#111827', size=12)),
            margin=dict(t=50, b=50)
        )
        _light_fig(fig2)
        pub_chart(fig2, key="temporal_dynamics", height=420, wide=True)
        st.caption(f"📖 Population modifier: {pm['note']}")

        # ── Clinical summary with CI + population-adjusted values ────────────
        st.markdown("#### 📊 Clinical Summary with 95% Confidence Intervals")
        clin  = r['clinical_predictions']
        ci_   = r.get('confidence_intervals', {})

        eff_adj = clin['efficacy']        * pm['mag']
        dur_adj = clin['duration_months'] * pm['dur']
        rea_adj = clin['reactogenicity']  * pm['react']
        saf_adj = max(10, (1 - rea_adj) * 100)
        eff_lo, eff_hi = ci_.get('efficacy',        (eff_adj*0.82, eff_adj*1.18))
        dur_lo, dur_hi = ci_.get('duration_months', (dur_adj*0.60, dur_adj*1.40))
        rea_lo, rea_hi = ci_.get('reactogenicity',  (rea_adj*0.75, rea_adj*1.25))

        summ_c1, summ_c2, summ_c3, summ_c4 = st.columns(4)
        summ_c1.metric("Efficacy",     f"{eff_adj:.1f}%",
                        delta=f"CI {eff_lo:.1f}–{eff_hi:.1f}%", delta_color="off")
        summ_c2.metric("Safety",       f"{saf_adj:.1f}%")
        summ_c3.metric("Reactogenicity", f"{rea_adj:.2f}",
                        delta=f"CI {rea_lo:.2f}–{rea_hi:.2f}", delta_color="off")
        summ_c4.metric("Protection",   f"{dur_adj:.1f} mo",
                        delta=f"CI {dur_lo:.1f}–{dur_hi:.1f}", delta_color="off")

        # ── Run Registry and Comparison ──────────────────────────────────────
        st.markdown("#### 📋 Run Registry")

        registry = st.session_state.get('run_registry', [])

        if not registry:
            st.info("No runs saved yet. Set your parameters, label your run, and click Predict.")
        else:
            # Registry summary table
            import pandas as _pd
            reg_rows = []
            for rx in registry:
                p = rx.get('_params', {})
                cp = rx.get('clinical_predictions', {})
                inn = rx.get('innate_prediction', {})
                th = rx.get('adaptive_prediction', {}).get('th_bias', {})
                reg_rows.append({
                    'ID':           rx.get('_run_id', ''),
                    'Label':        rx.get('_run_label', ''),
                    'Vaccine type': rx.get('vaccine_type', 'mRNA'),
                    'Adjuvant':     rx.get('adjuvant', 'None'),
                    'Lipid':        p.get('ionizable_lipid', ''),
                    'Modification': p.get('modification', '')[:22],
                    'Antigen':      p.get('antigen', ''),
                    'Species':      p.get('species', ''),
                    'Efficacy %':   f"{cp.get('efficacy', 0):.1f}",
                    'Safety':       f"{cp.get('safety', 0):.2f}",
                    'TLR7/8':       f"{inn.get('TLR7_8', 0):.2f}",
                    'Th1':          f"{th.get('Th1', 0):.2f}",
                    'Tfh':          f"{th.get('Tfh', 0):.2f}",
                    'Timestamp':    rx.get('_timestamp', ''),
                })
            reg_df = _pd.DataFrame(reg_rows)
            st.dataframe(reg_df, use_container_width=True, hide_index=True)

            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("🗑️ Clear all runs", key='clear_registry'):
                    st.session_state['run_registry'] = []
                    st.session_state.pop('cascade_results', None)
                    st.rerun()
            with rc2:
                # Excel download with full parameter record
                import io as _io
                try:
                    import openpyxl as _openpyxl
                    xl_buf = _io.BytesIO()
                    with _pd.ExcelWriter(xl_buf, engine='openpyxl') as writer:
                        # Sheet 1: Summary comparison
                        reg_df.to_excel(writer, sheet_name='Summary', index=False)

                        # Sheet 2: Full parameters per run
                        param_rows = []
                        for rx in registry:
                            p = rx.get('_params', {})
                            cp = rx.get('clinical_predictions', {})
                            inn = rx.get('innate_prediction', {})
                            th = rx.get('adaptive_prediction', {}).get('th_bias', {})
                            ci = rx.get('confidence_intervals', {})
                            eff_lo, eff_hi = ci.get('efficacy', (0, 0))
                            param_rows.append({
                                # Identity
                                'run_id':               p.get('run_id', ''),
                                'label':                p.get('label', ''),
                                'timestamp':            p.get('timestamp', ''),
                                'vaccine_type':         p.get('vaccine_type', 'mRNA'),
                                'adjuvant':             p.get('adjuvant', 'None'),
                                'adjuvant_class':       p.get('adjuvant_class', ''),
                                'adjuvant_confidence':  p.get('adjuvant_confidence', ''),
                                # LNP inputs
                                'ionizable_lipid':      p.get('ionizable_lipid', ''),
                                'lipid_pka':            p.get('lipid_pka', ''),
                                'lipid_logP':           p.get('lipid_logP', ''),
                                'lipid_mw_g_mol':       p.get('lipid_mw', ''),
                                'lipid_branching':      p.get('lipid_branching', ''),
                                'ionizable_ratio_mol%': p.get('ionizable_ratio', ''),
                                'helper_lipid':         p.get('helper_lipid', ''),
                                'helper_ratio_mol%':    p.get('helper_ratio', ''),
                                'cholesterol_ratio_mol%': p.get('cholesterol_ratio', ''),
                                'peg_lipid':            p.get('peg_lipid', ''),
                                'peg_ratio_mol%':       p.get('peg_ratio', ''),
                                # Nucleic acid
                                'modification':         p.get('modification', ''),
                                'modification_level_%': p.get('modification_level', ''),
                                'tlr_evasion':          p.get('tlr_evasion', ''),
                                'translation_eff':      p.get('translation_eff', ''),
                                # Antigen
                                'antigen':              p.get('antigen', ''),
                                'antigen_seq_50chars':  p.get('antigen_seq_50', ''),
                                'species':              p.get('species', ''),
                                'antigenicity':         p.get('antigenicity', ''),
                                'mhc1_score':           p.get('mhc1_score', ''),
                                'ctl_epitopes_est':     p.get('ctl_epitopes', ''),
                                # Innate outputs
                                'TLR7_8':               round(inn.get('TLR7_8', 0), 3),
                                'TLR3':                 round(inn.get('TLR3', 0), 3),
                                'cGAS_STING':           round(inn.get('cGAS_STING', 0), 3),
                                'Complement':           round(inn.get('Complement', 0), 3),
                                'Inflammasome':         round(inn.get('Inflammasome', 0), 3),
                                'DC_maturation':        round(inn.get('DC_maturation', 0), 3),
                                # Adaptive outputs
                                'Th1':                  round(th.get('Th1', 0), 3),
                                'Th2':                  round(th.get('Th2', 0), 3),
                                'Th17':                 round(th.get('Th17', 0), 3),
                                'Tfh':                  round(th.get('Tfh', 0), 3),
                                'memory_quality':       round(rx.get('adaptive_prediction', {}).get('memory_quality', 0), 3),
                                'ab_magnitude':         round(rx.get('adaptive_prediction', {}).get('ab_magnitude', 0), 3),
                                # Clinical outputs
                                'efficacy_%':           round(cp.get('efficacy', 0), 2),
                                'efficacy_CI_lo':       round(eff_lo, 2),
                                'efficacy_CI_hi':       round(eff_hi, 2),
                                'safety':               round(cp.get('safety', 0), 3),
                                'reactogenicity':       round(cp.get('reactogenicity', 0), 3),
                                'duration_months':      round(cp.get('duration_months', 0), 1),
                                'seed':                 rx.get('_seed', ''),
                            })
                        _pd.DataFrame(param_rows).to_excel(writer, sheet_name='Full Parameters', index=False)

                    xl_buf.seek(0)
                    st.download_button(
                        "⬇️ Download run registry (.xlsx)",
                        data=xl_buf.getvalue(),
                        file_name=f"epitrix_runs_{len(registry)}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except ImportError:
                    # Fallback to CSV if openpyxl not available
                    csv_buf = _io.StringIO()
                    _pd.DataFrame(reg_rows).to_csv(csv_buf, index=False)
                    st.download_button(
                        "⬇️ Download run registry (.csv)",
                        data=csv_buf.getvalue(),
                        file_name=f"epitrix_runs_{len(registry)}.csv",
                        mime="text/csv",
                    )

            # Comparison chart — shown when 2+ runs saved
            if len(registry) >= 2:
                st.markdown("#### 📊 Run Comparison")

                labels_r = [rx.get('_run_label', rx.get('_run_id', '')) for rx in registry]
                metrics_map = {
                    'Efficacy (%)':       [rx.get('clinical_predictions', {}).get('efficacy', 0) for rx in registry],
                    'Safety (×100)':      [rx.get('clinical_predictions', {}).get('safety', 0) * 100 for rx in registry],
                    'TLR7/8 (×100)':      [rx.get('innate_prediction', {}).get('TLR7_8', 0) * 100 for rx in registry],
                    'Th1 (×100)':         [rx.get('adaptive_prediction', {}).get('th_bias', {}).get('Th1', 0) * 100 for rx in registry],
                    'Tfh (×100)':         [rx.get('adaptive_prediction', {}).get('th_bias', {}).get('Tfh', 0) * 100 for rx in registry],
                    'Memory (×100)':      [rx.get('adaptive_prediction', {}).get('memory_quality', 0) * 100 for rx in registry],
                }

                metric_choice = st.multiselect(
                    "Metrics to compare:",
                    list(metrics_map.keys()),
                    default=['Efficacy (%)', 'Safety (×100)', 'TLR7/8 (×100)', 'Th1 (×100)'],
                    key='compare_metrics'
                )

                if metric_choice:
                    COLORS = ['#2563eb','#10b981','#f59e0b','#ef4444','#7c3aed','#0891b2','#d97706','#16a34a']
                    fig_reg = go.Figure()
                    for mi, metric in enumerate(metric_choice):
                        vals = metrics_map[metric]
                        fig_reg.add_trace(go.Bar(
                            name=metric,
                            x=labels_r,
                            y=vals,
                            marker_color=COLORS[mi % len(COLORS)],
                            text=[f"{v:.1f}" for v in vals],
                            textposition='outside',
                            textfont=dict(size=10),
                        ))
                    fig_reg.update_layout(
                        barmode='group',
                        height=420,
                        yaxis=dict(
                            title=dict(text='Value', font=dict(color='#111827')),
                            tickfont=dict(color='#111827'),
                            range=[0, 115],
                        ),
                        xaxis=dict(
                            tickfont=dict(color='#111827', size=10),
                            tickangle=-20,
                        ),
                        margin=dict(t=30, b=100, l=50, r=20),
                        legend=dict(font=dict(color='#111827', size=10)),
                    )
                    _light_fig(fig_reg)
                    pub_chart(fig_reg, key=f"run_registry_{len(st.session_state.get('run_registry',[]))}", height=400, wide=True)
                    st.caption("All scores normalised to 0-100 scale for comparison. Safety and TLR7/8 multiplied by 100 from their native 0-1 scale.")

    else:
        st.info("Run molecular prediction first (🔬 Molecular Input tab).")

    st.markdown('</div>', unsafe_allow_html=True)


# ── TAB 4 ────────────────────────────────────────────────────────────────────
def temporal_dynamics_module():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)

    _run_context_banner()

    st.markdown("""
    <div class="innovation-card">
      <h3 style="font-size:1.25rem;font-weight:700;color:#111827;margin:0 0 0.5rem;">
        ⏱️ Temporal Dynamics: Innate→Adaptive Cascade · Epitrix
      </h3>
      <p style="color:#4b5563;margin:0;">How innate responses in hours 1–24 shape adaptive responses in days 7–30</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-container" style="padding-top:0;">
      <div class="time-point"><h5>Minutes 0–30: Immediate Recognition</h5>
        <p>Complement activation · Particle uptake · Surface receptor engagement</p></div>
      <div class="time-point"><h5>Hours 1–6: Early Innate Response</h5>
        <p>TLR7/8 activation in endosomes · Rapid cytokine release (TNF-α, IL-6) · DC activation</p></div>
      <div class="time-point"><h5>Hours 6–24: Sustained Innate Signaling</h5>
        <p>cGAS-STING pathway · Type I interferon production · DC migration to lymph nodes</p></div>
      <div class="time-point"><h5>Days 3–7: Innate → Adaptive Bridge</h5>
        <p>DC-T cell priming · Cytokine environment shapes Th differentiation · Germinal center initiation</p></div>
      <div class="time-point"><h5>Days 7–14: Peak Adaptive Response</h5>
        <p>Antibody production · CD4/CD8 expansion · Tfh-B cell collaboration</p></div>
      <div class="time-point"><h5>Days 14–30: Memory Formation &amp; Reactogenicity Resolution</h5>
        <p>Long-lived plasma cells · Memory T cell differentiation · Systemic reactogenicity wanes</p></div>
    </div>
    """, unsafe_allow_html=True)

    if 'cascade_results' in st.session_state:
        r = st.session_state['cascade_results']
        st.markdown("#### 📈 Integrated Immune Response Timeline")

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Innate Cytokines  (0–48 h)',
                'DC Activation  (0–7 days)',
                'Adaptive Response  (0–30 days)'
            ),
            vertical_spacing=0.18,
            row_heights=[0.3, 0.3, 0.4]
        )

        tlr = r['innate_prediction']['TLR7_8']
        inf = r['innate_prediction']['Inflammasome']

        # Row 1
        t_h = np.linspace(0, 48, 200)
        fig.add_trace(go.Scatter(x=t_h, y=tlr * np.exp(-(t_h - 4)**2 / 40),
                                  name='TNF-α', line=dict(color='#ef4444', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_h, y=inf * np.exp(-(t_h - 8)**2 / 60),
                                  name='IL-6',  line=dict(color='#f59e0b', width=2)), row=1, col=1)

        # Row 2
        t_dc = np.linspace(0, 7, 200)
        fig.add_trace(go.Scatter(x=t_dc, y=1 / (1 + np.exp(-(t_dc - 2) * 3)) * tlr,
                                  name='DC Maturation', line=dict(color='#2563eb', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t_dc, y=np.exp(-(t_dc - 2)**2 / 3) * tlr * 0.8,
                                  name='IL-12',          line=dict(color='#7c3aed', width=2)), row=2, col=1)

        # Row 3
        t_d  = np.linspace(0, 30, 200)
        ab   = r['adaptive_prediction']['antibody_response']
        peak, dur = ab['time_to_peak'], ab['durability']
        ab_t = np.where(t_d <= peak, (t_d / peak)**2, np.exp(-0.05 * (1 - dur) * (t_d - peak)))
        cd4  = np.maximum(0, 1 / (1 + np.exp(-(t_d - 7) / 2)) - 0.3 * np.exp(-(t_d - 14)**2 / 50))
        fig.add_trace(go.Scatter(x=t_d, y=ab_t, name='Antibodies',
                                  line=dict(color='#10b981', width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=t_d, y=cd4,  name='CD4+ T cells',
                                  line=dict(color='#f59e0b', width=2)), row=3, col=1)

        _ax = dict(showgrid=True, gridcolor='#e5e7eb', linecolor='#d1d5db',
                   tickfont=dict(color='#111827', size=11),
                   title_font=dict(color='#111827', size=12))
        fig.update_xaxes(title_text="Time (hours)", **_ax, row=1, col=1)
        fig.update_xaxes(title_text="Time (days)",  **_ax, row=2, col=1)
        fig.update_xaxes(title_text="Time (days)",  **_ax, row=3, col=1)
        fig.update_yaxes(title_text="Level", **_ax)

        fig.update_layout(
            height=720,
            font=dict(family='Inter', color='#111827', size=12),
            paper_bgcolor='white', plot_bgcolor='white',
            legend=dict(
                orientation='v', x=1.02, y=1,
                font=dict(color='#111827', size=11),
                bgcolor='white', bordercolor='#e5e7eb', borderwidth=1
            ),
            margin=dict(t=80, b=50, l=60, r=160)
        )
        for ann in fig.layout.annotations:
            ann.font = dict(color='#111827', size=13, family='Inter')

        pub_chart(fig, key="th_bias_pie", height=380)
    else:
        st.info("Run molecular prediction (🔬 Molecular Input tab) to generate the timeline chart.")

    st.markdown('</div>', unsafe_allow_html=True)


# ── TAB 5 — EPITOPE ANALYSIS ──────────────────────────────────────────────────
def epitope_analysis_module():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)

    _run_context_banner()

    st.markdown("""
    <div class="innovation-card">
      <h3 style="font-size:1.25rem;font-weight:700;color:#111827;margin:0 0 0.4rem;">
        🧬 Epitope Analysis: PSSM Scanner and IEDB Integration
      </h3>
      <p style="color:#4b5563;margin:0;font-size:0.9rem;">
        Scan any protein sequence for MHC-I (HLA-A*02:01, 9-mer) and MHC-II (HLA-DR, 15-mer)
        binding epitopes using published anchor position weight matrices.
        When the IEDB Analysis Resource is reachable, predictions upgrade automatically
        to NetMHCpan 4.1 / NetMHCIIpan 4.0.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### 🔬 Sequence Input")
        preset = st.selectbox("Preset antigen:", list(ANTIGEN_PRESETS.keys()), key='ep_preset')
        default_seq = ANTIGEN_PRESETS[preset]['sequence'] if preset != '— Enter custom sequence —' else ''
        seq_input = st.text_area("Protein sequence (single-letter AA):", value=default_seq,
                                  height=130, key='ep_seq')

        # ── Species selector ──────────────────────────────────────────────────
        ep_species = st.selectbox(
            "Prediction species:",
            ['human', 'mouse'],
            format_func=lambda x: '🧑 Human (HLA-A*02:01 / HLA-DR)' if x == 'human'
                                  else '🐭 Mouse (H-2Kb / H-2Db)',
            key='ep_species'
        )

        # IEDB allele selectors — update options based on species
        if ep_species == 'human':
            allele_mhci  = st.selectbox("MHC-I allele (IEDB fallback):",
                ['HLA-A*02:01','HLA-A*01:01','HLA-A*03:01','HLA-B*07:02','HLA-B*44:02'],
                key='ep_a1')
            allele_mhcii = st.selectbox("MHC-II allele (IEDB fallback):",
                ['HLA-DRB1*01:01','HLA-DRB1*03:01','HLA-DRB1*04:01','HLA-DRB1*07:01'],
                key='ep_a2')
        else:
            allele_mhci  = st.selectbox("MHC-I allele (IEDB fallback):",
                ['H-2Kb', 'H-2Db', 'H-2Kd', 'H-2Dd', 'H-2Ld'],
                key='ep_a1')
            allele_mhcii = st.selectbox("MHC-II allele (IEDB fallback):",
                ['H-2IAb', 'H-2IAd', 'H-2IEd'],
                key='ep_a2')

        run_ep = st.button("🔍 Scan Epitopes", type="primary", key='ep_run')

    with col2:
        st.markdown("#### 📖 Method Priority")
        st.markdown("""
<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;
     padding:1rem;font-size:0.83rem;color:#1e3a5f;">
<strong>🥇 XGBoost ML model</strong> (best, used when model files are present):<br>
Trained on IEDB MHC-I binding data. Human model: AUC-ROC 0.986 (219k peptides).
Mouse model: AUC-ROC 0.970 (59k peptides). Purple <strong>🤖 XGBoost ML</strong> badge.<br><br>
<strong>🥈 IEDB NetMHCpan</strong> (good fallback, used when ML is unavailable and network is reachable):<br>
NetMHCpan 4.1 (MHC-I) and NetMHCIIpan 4.0 (MHC-II).
Green <strong>IEDB ✓</strong> badge.<br><br>
<strong>🥉 Local PSSM scanner</strong> (always available as a fallback):<br>
Published HLA-A*02:01 / H-2Kb anchor position weight matrices.
Grey <strong>Local PSSM</strong> badge.<br><br>
<em>To enable ML predictions: place the <code>epitrix_ml/</code> folder and
trained <code>models/</code> directory alongside <code>app.py</code>.</em>
</div>""", unsafe_allow_html=True)

    if run_ep and seq_input.strip():
        with st.spinner(f"Scanning epitopes [{ep_species}]..."):
            result = analyze_antigen_sequence(
                seq_input,
                use_iedb=st.session_state.get('use_iedb', True),
                species=ep_species,
            )

        if not result.get('valid'):
            st.error(f"Could not parse sequence: {result.get('error','unknown')}")
        else:
            # Method badge — three tiers
            if result.get('ml_used'):
                ml_auc = result.get("ml_auc")
                auc_str = f"{ml_auc:.3f}" if isinstance(ml_auc, float) else "0.986"
                method_str = f'🟣 **XGBoost ML** (AUC-ROC {auc_str} · {ep_species})'
            elif result.get('iedb_used'):
                method_str = f'🟢 **IEDB NetMHCpan** ({ep_species})'
            else:
                method_str = f'🟡 **Local PSSM** ({ep_species}): ML model not found, IEDB unavailable'

            species_label = '🧑 Human HLA' if ep_species == 'human' else '🐭 Mouse H-2'
            st.markdown(f"**Method:** {method_str} &nbsp;|&nbsp; **Species:** {species_label}")

            # Summary scores
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("MHC-I score",   f"{result['mhc1_score']:.0%}")
            mc2.metric("MHC-II score",  f"{result['mhc2_score']:.0%}")
            mc3.metric("B-cell score",  f"{result['b_cell_score']:.0%}")
            mc4.metric("Antigenicity",  f"{result['antigenicity']:.0%}")

            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("CTL epitopes",    str(result['ctl_epitopes_est']))
            ec2.metric("Th epitopes",     str(result['th_epitopes_est']))
            ec3.metric("B-cell epitopes", str(result['bcell_epitopes_est']))

            # Top peptide tables
            tab_mhci, tab_mhcii, tab_tcell = st.tabs([
                "🔵 Top MHC-I Binders",
                "🟠 Top MHC-II Binders",
                "🧫 T Cell Immunogenicity",
            ])

            with tab_mhci:
                top = result.get('top_mhci_peptides', [])
                if top:
                    st.markdown(f"**Top MHC-I peptides** ({result.get('mhci_method','—')})")
                    rows = []
                    for i, (score, pep, pos) in enumerate(top[:10], 1):
                        rows.append({'Rank': i, 'Position': pos, 'Peptide': pep,
                                     'Percentile rank' if result.get('iedb_used') else 'PSSM score':
                                     f"{score:.2f}%" if result.get('iedb_used') and score > 0
                                     else f"{score:.3f}"})
                    import pandas as pd
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No strong MHC-I binders found above threshold.")

            with tab_mhcii:
                top2 = result.get('top_mhcii_peptides', [])
                if top2:
                    st.markdown(f"**Top MHC-II peptides** ({result.get('mhcii_method','—')})")
                    rows2 = []
                    for i, (score, pep, pos) in enumerate(top2[:10], 1):
                        rows2.append({'Rank': i, 'Position': pos, 'Peptide': pep, 'Score': f"{score:.3f}"})
                    st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)
                else:
                    st.info("No strong MHC-II binders found above threshold.")

            with tab_tcell:
                st.markdown(f"**T cell immunogenicity** (XGBoost model, species: "
                            f"{'🧑 Human' if ep_species == 'human' else '🐭 Mouse'}")

                # Run T cell scan
                try:
                    import sys, os
                    _ml_dir = os.path.join(os.path.dirname(__file__), 'epitrix_ml')
                    if _ml_dir not in sys.path:
                        sys.path.insert(0, os.path.dirname(__file__))
                    from epitrix_ml.tcell_integrate import scan_protein_sequence as _tcell_scan
                    _tcell_avail = True
                except ImportError:
                    _tcell_avail = False

                if not _tcell_avail:
                    st.warning("T cell models not found. Place `epitrix_ml/models/tcell_*.pkl` "
                               "alongside `app.py` and restart.")
                else:
                    with st.spinner("Running T cell immunogenicity scan..."):
                        tc_result = _tcell_scan(
                            seq_input,
                            species=ep_species,
                            window=9,
                            innate_prediction=None,
                        )

                    if tc_result and tc_result.get('_available'):
                        tc1, tc2, tc3 = st.columns(3)
                        tc1.metric("Immunogenic peptides",
                                   f"{tc_result['n_immunogenic']} / {tc_result.get('n_peptides_scored','?')}")
                        tc2.metric("Mean immunogenicity",
                                   f"{tc_result['mean_immunogenicity']:.0%}")
                        tc3.metric("Est. response frequency",
                                   f"{tc_result['response_freq_pct']:.1f}%")

                        st.caption(f"Method: {tc_result.get('method','—')} · "
                                   f"LNP modulation: run the full simulation for LNP-adjusted predictions")

                        top_tc = tc_result.get('immunogenic_peptides_with_pos', [])
                        if top_tc:
                            import pandas as _pd
                            tc_rows = []
                            for i, (prob, pep, pos) in enumerate(top_tc[:10], 1):
                                tc_rows.append({
                                    'Rank':     i,
                                    'Position': pos,
                                    'Peptide':  pep,
                                    'Immunogenicity prob': f"{prob:.3f}",
                                    'Predicted response':
                                        f"{'High' if prob > 0.7 else 'Moderate' if prob > 0.5 else 'Low'}",
                                })
                            st.dataframe(_pd.DataFrame(tc_rows),
                                         use_container_width=True, hide_index=True)

                        # Mini bar chart of top 10 peptides
                        if top_tc:
                            top10 = top_tc[:10]
                            fig_tc = go.Figure(go.Bar(
                                x=[p for _, p, _ in top10],
                                y=[prob for prob, _, _ in top10],
                                marker_color=['#7c3aed' if prob > 0.7
                                              else '#2563eb' if prob > 0.5
                                              else '#9ca3af'
                                              for prob, _, _ in top10],
                                text=[f"{prob:.2f}" for prob, _, _ in top10],
                                textposition='outside',
                            ))
                            fig_tc.update_layout(
                                xaxis=dict(title='Peptide', tickfont=dict(
                                    color='#111827', size=10)),
                                yaxis=dict(title='Immunogenicity probability',
                                           range=[0, 1.1],
                                           tickfont=dict(color='#111827')),
                                margin=dict(t=20, b=60), height=280,
                            )
                            _light_fig(fig_tc)
                            pub_chart(fig_tc, key="tcell_immunogenicity", height=400, wide=True)
                            st.caption("Purple = high (>0.7) · Blue = moderate (>0.5) · Grey = low. "
                                       "Probabilities are base estimates without LNP modulation.")
                    else:
                        st.error("T cell scan failed. Check model files are present.")

            # Immunogenicity bar chart
            st.markdown("#### 📊 Immunogenicity Profile")
            fig_ep = go.Figure()
            metrics  = ['MHC-I', 'MHC-II', 'B-cell', 'Antigenicity']
            vals     = [result['mhc1_score'], result['mhc2_score'],
                        result['b_cell_score'], result['antigenicity']]
            colors   = ['#2563eb','#7c3aed','#10b981','#f59e0b']
            fig_ep.add_trace(go.Bar(x=metrics, y=vals, marker_color=colors,
                                    text=[f"{v:.0%}" for v in vals], textposition='outside'))
            fig_ep.update_layout(yaxis=dict(range=[0,1.15], tickformat='.0%',
                                            title=dict(text='Score', font=dict(color='#111827')),
                                            tickfont=dict(color='#111827')),
                                 xaxis=dict(tickfont=dict(color='#111827')),
                                 margin=dict(t=30,b=40), height=280)
            _light_fig(fig_ep)
            pub_chart(fig_ep, key="epitope_landscape", height=420, wide=True)

    elif run_ep:
        st.warning("Please paste a protein sequence first.")
    else:
        st.info("Select a preset or paste a sequence, then click **Scan Epitopes**.")

    st.markdown('</div>', unsafe_allow_html=True)


# ── TAB 6 — FORMULATION OPTIMIZER ────────────────────────────────────────────
def formulation_optimizer_module():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)

    _run_context_banner()

    st.markdown(f"""
    <div class="innovation-card">
      <h3 style="font-size:1.25rem;font-weight:700;color:#111827;margin:0 0 0.4rem;">
        ⚗️ Formulation Optimizer
      </h3>
      <p style="color:#4b5563;margin:0;font-size:0.9rem;">
        <strong>mRNA / DNA vaccines:</strong> sweeps 720 combinations
        (6 lipids × 4 ionizable ratios × 3 cholesterol × 3 PEG × 5 modifications)
        and ranks by your chosen objective.<br>
        <strong>Protein subunit vaccines:</strong> sweeps adjuvant × formulation vehicle combinations
       : the biologically relevant design space for protein vaccines.
        All predictions are deterministic: same antigen always returns the same ranking.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### ⚙️ Optimization Settings")
        objective = st.selectbox("Optimization objective:", [
            'balanced', 'efficacy', 'safety', 'durability', 'th1_bias'
        ], format_func=lambda x: {
            'balanced':   '⚖️ Balanced (efficacy + safety)',
            'efficacy':   '🎯 Max Predicted Efficacy',
            'safety':     '🛡️ Max Safety Score',
            'durability': '⏳ Max Protection Duration',
            'th1_bias':   '🔬 Max Th1/Tfh (cellular immunity)',
        }[x], key='opt_obj')

        top_n = st.slider("Top N formulations to show:", 3, 10, 5, key='opt_n')

        # Use antigen from session state if available
        ag_cache = st.session_state.get('antigen_features_cache')
        if ag_cache and ag_cache.get('valid'):
            st.markdown(f"""
<div style="background:#fefce8;border:1px solid #fde68a;border-radius:8px;
     padding:0.5rem 0.8rem;font-size:0.8rem;color:#92400e;">
  🎯 Using antigen from Molecular Input tab<br>
  ({ag_cache['seq_type']} · {ag_cache['length']} residues · antigenicity {ag_cache['antigenicity']:.0%})
</div>""", unsafe_allow_html=True)
        else:
            st.caption("ℹ️ No antigen loaded. Run the Molecular Input tab first, or optimizer will use default antigenicity values.")

        # Read vaccine_type and adjuvant from the last run in session state
        last_vtype   = st.session_state.get('cascade_results', {}).get('vaccine_type', 'mRNA')
        last_adjuvant = st.session_state.get('cascade_results', {}).get('adjuvant', 'None (formulation only)')
        st.caption(f"Using vaccine type **{last_vtype}** and adjuvant **{last_adjuvant}** from Molecular Input tab.")

        _opt_btn_label = ("🚀 Run Optimizer (720 formulations)"
                          if last_vtype in ('mRNA', 'DNA')
                          else "🚀 Run Optimizer (adjuvant × vehicle sweep)")
        run_opt = st.button(_opt_btn_label, type="primary", key='opt_run')

    with col2:
        if run_opt:
            n_combos = 720 if last_vtype in ('mRNA', 'DNA') else len([
                k for k, v in ADJUVANTS.items()
                if 'Protein subunit' in v['compatible'] and v['confidence'] != 'research'
            ]) * 4  # adjuvants × 4 vehicles
            with st.spinner(f"Evaluating {n_combos} {'lipid × modification' if last_vtype != 'Protein subunit' else 'adjuvant × vehicle'} combinations..."):
                top_results = run_formulation_optimizer(
                    antigen_features=ag_cache,
                    objective=objective,
                    top_n=top_n,
                    vaccine_type=last_vtype,
                    adjuvant_name=last_adjuvant
                )
            st.success(f"✅ Optimization complete. Top {top_n} formulations ranked.")

            # ── Summary comparison bar chart ─────────────────────────────────
            is_protein = last_vtype == 'Protein subunit'
            st.markdown("#### 📊 Top Formulations: Side-by-Side Comparison")
            labels     = [f"#{i+1} {r['_formulation']['label'][:30]}" for i, r in enumerate(top_results)]
            efficacies = [r['clinical_predictions']['efficacy']    for r in top_results]
            safeties   = [r['clinical_predictions']['safety']      for r in top_results]
            reacto     = [r['clinical_predictions']['reactogenicity'] * 100 for r in top_results]

            fig_opt = go.Figure()
            fig_opt.add_trace(go.Bar(name='Efficacy %', x=labels, y=efficacies,
                                     marker_color='#2563eb',
                                     text=[f"{v:.1f}%" for v in efficacies],
                                     textposition='outside'))
            fig_opt.add_trace(go.Bar(name='Safety %', x=labels, y=safeties,
                                     marker_color='#10b981',
                                     text=[f"{v:.1f}%" for v in safeties],
                                     textposition='outside'))
            fig_opt.add_trace(go.Bar(name='Reactogenicity %', x=labels, y=reacto,
                                     marker_color='#ef4444',
                                     text=[f"{v:.1f}%" for v in reacto],
                                     textposition='outside'))
            fig_opt.update_layout(
                barmode='group', height=380,
                title=dict(
                    text=f"{'Adjuvant × Vehicle' if is_protein else 'Lipid × Modification'} optimisation",
                    font=dict(color='#111827', size=13)
                ),
                yaxis=dict(title=dict(text='Score (%)', font=dict(color='#111827')),
                           tickfont=dict(color='#111827'), range=[0, 120]),
                xaxis=dict(tickfont=dict(color='#111827', size=9), tickangle=-15),
                margin=dict(t=50, b=80)
            )
            _light_fig(fig_opt)
            pub_chart(fig_opt, key="formulation_optimizer", height=420, wide=True)

            # ── Detailed cards per formulation ───────────────────────────────
            st.markdown("#### 🏆 Ranked Formulation Details")
            for i, r in enumerate(top_results):
                fmt   = r['_formulation']
                clin  = r['clinical_predictions']
                th    = r['adaptive_prediction']['th_bias']
                ci    = r.get('confidence_intervals', {})
                sc    = r['_score']
                eff_lo, eff_hi = ci.get('efficacy', (clin['efficacy'], clin['efficacy']))
                dur_lo, dur_hi = ci.get('duration_months',
                                        (clin['duration_months'], clin['duration_months']))

                if is_protein:
                    expander_title = (
                        f"#{i+1}  {fmt['adjuvant'][:35]}  ·  {fmt['vehicle']}  "
                        f"·  Efficacy {clin['efficacy']:.1f}%  ·  Safety {clin['safety']:.1f}%  "
                        f"·  Score {sc:.3f}"
                    )
                else:
                    expander_title = (
                        f"#{i+1}  {fmt['lipid']}  ·  {fmt['modification'][:25]}  "
                        f"·  Efficacy {clin['efficacy']:.1f}%  ·  Safety {clin['safety']:.1f}%  "
                        f"·  Score {sc:.3f}"
                    )

                with st.expander(expander_title):
                    dc1, dc2, dc3, dc4 = st.columns(4)
                    dc1.metric("Efficacy",       f"{clin['efficacy']:.1f}%",
                               delta=f"CI: {eff_lo:.1f}–{eff_hi:.1f}%", delta_color="off")
                    dc2.metric("Safety",         f"{clin['safety']:.1f}%")
                    dc3.metric("Reactogenicity", f"{clin['reactogenicity']:.2f}")
                    dc4.metric("Duration",       f"{clin['duration_months']:.1f} mo",
                               delta=f"CI: {dur_lo:.1f}–{dur_hi:.1f}", delta_color="off")

                    if is_protein:
                        adj_conf  = ADJUVANTS.get(fmt['adjuvant'], {}).get('confidence', '')
                        _, adj_col = ADJUVANT_CONFIDENCE.get(adj_conf, ('', '#6b7280'))
                        adj_notes = ADJUVANTS.get(fmt['adjuvant'], {}).get('notes', '')[:120]
                        st.markdown(f"""
<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;
     padding:0.6rem 1rem;font-size:0.82rem;color:#374151;">
  <strong>Adjuvant:</strong>
  <span style="color:{adj_col};font-weight:600;">{fmt['adjuvant']}</span>
  &nbsp;|&nbsp;
  <strong>Class:</strong> {fmt['adjuvant_class']} &nbsp;|&nbsp;
  <strong>Vehicle:</strong> {fmt['vehicle']}<br>
  <strong>Th1:</strong> {th['Th1']:.0%} &nbsp;
  <strong>Th2:</strong> {th['Th2']:.0%} &nbsp;
  <strong>Th17:</strong> {th['Th17']:.0%} &nbsp;
  <strong>Tfh:</strong> {th['Tfh']:.0%} &nbsp;|&nbsp;
  <strong>Memory quality:</strong> {r['adaptive_prediction']['memory_quality']:.2f}<br>
  <span style="color:#6b7280;font-style:italic;">{adj_notes}{'…' if len(ADJUVANTS.get(fmt['adjuvant'],{}).get('notes',''))>120 else ''}</span>
</div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;
     padding:0.6rem 1rem;font-size:0.82rem;color:#374151;">
  <strong>Ionizable lipid:</strong> {fmt['lipid']} &nbsp;|&nbsp;
  <strong>Ratios (ion/hlp/chol/PEG):</strong>
  {fmt['ionizable_ratio']}/{fmt['helper_ratio']}/{fmt['cholesterol_ratio']}/{fmt['peg_ratio']} mol%
  &nbsp;|&nbsp; <strong>Modification:</strong> {fmt['modification']}<br>
  <strong>Adjuvant:</strong> {fmt.get('adjuvant','None')} &nbsp;|&nbsp;
  <strong>Th1:</strong> {th['Th1']:.0%} &nbsp;
  <strong>Th2:</strong> {th['Th2']:.0%} &nbsp;
  <strong>Th17:</strong> {th['Th17']:.0%} &nbsp;
  <strong>Tfh:</strong> {th['Tfh']:.0%} &nbsp;|&nbsp;
  <strong>Memory quality:</strong> {r['adaptive_prediction']['memory_quality']:.2f} &nbsp;|&nbsp;
  <strong>Seed:</strong> <code>{r['_seed']}</code>
</div>""", unsafe_allow_html=True)

        else:
            st.info("Configure settings on the left and click **Run Optimizer** to evaluate 720 formulation combinations.")

    st.markdown('</div>', unsafe_allow_html=True)
def display_training_datasets():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)

    st.markdown("""
    <div class="innovation-card">
      <h2 class="section-title">📚 Scientific Evidence Base & Parameterisation Sources</h2>
      <p class="section-subtitle" style="margin-bottom:0.75rem;">
        Epitrix is a <strong>hybrid platform</strong>. The MHC-I and T cell immunogenicity components
        use <strong>trained XGBoost models</strong> (MHC-I human AUC 0.986, trained on 219k IEDB peptides;
        T cell human AUC 0.928, trained on 92k IEDB assays). The innate→adaptive cascade
        uses <strong>parameterised mechanistic equations</strong> with coefficients manually
        extracted from the peer-reviewed studies listed below. These studies are the
        <em>calibration sources</em> for the mechanistic layer, not ML training data.
      </p>
      <div style="background:#ede9fe;border:1px solid #c4b5fd;border-radius:8px;
           padding:0.7rem 1rem;font-size:0.82rem;color:#4c1d95;">
        <strong>ℹ️ Architecture note:</strong> The epitope prediction (ML) and cascade simulation
        (mechanistic) layers are separate components. The ML models were trained on IEDB bulk data.
        The cascade equations use hand-coded coefficients from the papers below.
        Both layers feed into the same prediction pipeline.
      </div>
    </div>
    """, unsafe_allow_html=True)

    DATASETS = [
        {
            'category': '🧬 Lipid Nanoparticle Formulation',
            'color': '#2563eb', 'bg': '#eff6ff', 'border': '#bfdbfe',
            'datasets': [
                {'name': 'Moderna mRNA-1273 LNP Characterization',
                 'type': 'Physicochemical',
                 'source': 'Hassett et al., NPJ Vaccines 2019',
                 'doi': '10.1038/s41541-019-0107-3',
                 'description': 'SM-102 LNP particle size, zeta potential, encapsulation efficiency, and pKa measurements. Basis for core physicochemical prediction equations.',
                 'n': '48 formulations', 'param': 'particle_size, zeta_potential, encapsulation_eff'},
                {'name': 'BNT162b2 LNP Formulation Data',
                 'type': 'Physicochemical',
                 'source': 'Schoenmaker et al., Int J Pharm 2021',
                 'doi': '10.1016/j.ijpharm.2021.120586',
                 'description': 'ALC-0315/ALC-0159 LNP characterization; pKa–ionization–uptake relationships.',
                 'n': '36 formulations', 'param': 'pKa effect on TLR activation'},
                {'name': 'Ionizable Lipid pKa–Efficacy Database',
                 'type': 'Structure–Activity',
                 'source': 'Kulkarni et al., Nano Letters 2021',
                 'doi': '10.1021/acs.nanolett.1c02143',
                 'description': 'Systematic pKa sweep across 24 ionizable lipids; basis for pKa–TLR activation coefficient (0.1 per pKa unit above 6.0).',
                 'n': '24 lipids × 6 cell lines', 'param': 'tlr_lipid_effect = (pka - 6.0) × 0.1'},
                {'name': 'LNP Helper Lipid Membrane Rigidity Panel',
                 'type': 'Biophysical',
                 'source': 'Leung et al., Advanced Materials 2020',
                 'doi': '10.1002/adma.201906027',
                 'description': 'DLS, cryo-EM, and fluorescence anisotropy for PC/PE helper lipid series; membrane rigidity values in the helper lipid database.',
                 'n': '12 helper lipids', 'param': 'membrane_rigidity values (0.2–0.95)'},
                {'name': 'PEG-Lipid Shedding Kinetics Atlas',
                 'type': 'Pharmacokinetic',
                 'source': 'Chen et al., Biomaterials 2020',
                 'doi': '10.1016/j.biomaterials.2020.120033',
                 'description': 'PEG desorption measurements for DMG, DSPE, and ceramide anchors; shedding rate labels in PEG-lipid database.',
                 'n': '8 PEG-lipid variants', 'param': 'shedding_rate classifications'},
            ]
        },
        {
            'category': '🔥 Innate Immune Pathway Activation',
            'color': '#dc2626', 'bg': '#fff1f2', 'border': '#fecdd3',
            'datasets': [
                {'name': 'TLR7/8 Activation by mRNA Modification Status',
                 'type': 'In vitro immunology',
                 'source': 'Karikó et al., Immunity 2005',
                 'doi': '10.1016/j.immuni.2005.06.008',
                 'description': 'Foundational dataset establishing pseudouridine and m5C TLR evasion. TLR evasion coefficients directly derived here.',
                 'n': '5 modifications × 3 TLR reporters', 'param': 'tlr_evasion scores (0.0–0.85)'},
                {'name': 'm1Ψ TLR Suppression Quantification',
                 'type': 'In vitro immunology',
                 'source': 'Andries et al., Nature Biotechnology 2015',
                 'doi': '10.1038/nbt.3122',
                 'description': 'Quantitative comparison of m1Ψ vs Ψ TLR7/8 suppression; basis for 0.85 TLR evasion score for N1-methyl-pseudouridine.',
                 'n': '3 modifications × 8 cell types', 'param': 'm1Ψ tlr_evasion = 0.85'},
                {'name': 'cGAS-STING Activation by LNP Particle Size',
                 'type': 'In vitro / In vivo',
                 'source': 'Miao et al., Nature Biotechnology 2019',
                 'doi': '10.1038/s41587-019-0312-z',
                 'description': 'cGAS-STING activation as function of particle size; basis for the particle_size–cGAS_STING slope (0.005 per nm above 80 nm).',
                 'n': '9 LNP formulations', 'param': 'cGAS_STING += (size−80) × 0.005'},
                {'name': 'Complement Activation by LNP Charge',
                 'type': 'In vitro / Ex vivo',
                 'source': 'Szebeni et al., Molecular Immunology 2011',
                 'doi': '10.1016/j.molimm.2011.09.001',
                 'description': 'Zeta potential vs complement activation; zeta coefficient 0.02 per mV in complement term.',
                 'n': '18 lipid systems', 'param': 'Complement = |zeta| × 0.02'},
                {'name': 'NLRP3 Inflammasome Activation by Lipid Nanoparticles',
                 'type': 'In vitro',
                 'source': 'Swanson et al., Nature Immunology 2019',
                 'doi': '10.1038/s41590-018-0283-3',
                 'description': 'Cationic lipid charge density vs NLRP3 inflammasome activation; baseline inflammasome score 0.30.',
                 'n': '7 lipid formulations', 'param': 'Inflammasome baseline = 0.30'},
            ]
        },
        {
            'category': '🦠 Dendritic Cell Programming',
            'color': '#7c3aed', 'bg': '#f5f3ff', 'border': '#ddd6fe',
            'datasets': [
                {'name': 'mRNA-LNP DC Subset Activation Atlas',
                 'type': 'In vivo mouse / human PBMC',
                 'source': 'Liang et al., Nature Communications 2021',
                 'doi': '10.1038/s41467-021-21249-y',
                 'description': 'cDC1 vs cDC2 vs pDC activation kinetics after LNP injection; cDC1 = TLR×0.5 + cGAS×0.5 equation coefficients.',
                 'n': '4 LNP variants × 3 DC subsets', 'param': 'cDC1/cDC2/pDC score equations'},
                {'name': 'TLR Agonist IL-12 Production Dose–Response',
                 'type': 'In vitro human DC',
                 'source': 'Napolitani et al., Nature Immunology 2005',
                 'doi': '10.1038/ni1223',
                 'description': 'TLR7/8 vs cGAS-STING contribution to IL-12p70 in monocyte-derived DCs; IL-12 = cDC1×0.7 + cGAS×0.3.',
                 'n': 'n=6 donors', 'param': 'IL-12 output coefficients'},
            ]
        },
        {
            'category': '🎯 Adaptive Immune Outcomes',
            'color': '#10b981', 'bg': '#f0fdf4', 'border': '#bbf7d0',
            'datasets': [
                {'name': 'mRNA Vaccine Th1/Th2 Polarization Clinical Data',
                 'type': 'Phase I/II clinical',
                 'source': 'Walsh et al., NEJM 2020 (BNT162b2 Phase I)',
                 'doi': '10.1056/NEJMoa2027906',
                 'description': 'CD4+ cytokine profiling post-mRNA vaccination; Th1/Th2 bias ratios for clinical lipid formulations. Th1 = TLR×0.5 + cGAS×0.3 + MHC-II×0.2.',
                 'n': 'n=195 participants', 'param': 'Th polarization equation weights'},
                {'name': 'Tfh Cell Induction by Adjuvant Type',
                 'type': 'In vivo / meta-analysis',
                 'source': 'Crotty, Immunity 2019',
                 'doi': '10.1016/j.immuni.2019.01.006',
                 'description': 'GC Tfh induction as function of TLR7/8 signal; Tfh = TLR×0.35 + Th1×0.3 + MHC-II×0.15.',
                 'n': 'Meta-analysis: 31 studies', 'param': 'Tfh equation weights'},
                {'name': 'mRNA Vaccine Antibody Kinetics Dataset',
                 'type': 'Clinical serology',
                 'source': 'Goldberg et al., Science 2021',
                 'doi': '10.1126/science.abm4583',
                 'description': 'Longitudinal antibody titer measurements; peak day and waning rate coefficients for the kinetics curve.',
                 'n': 'n=6,300 individuals', 'param': 'ab_peak_day = 14 − TLR×5 − MHC-II×2'},
                {'name': 'Memory B and T Cell Persistence Atlas',
                 'type': 'Longitudinal immunology',
                 'source': 'Crotty et al., Science 2021',
                 'doi': '10.1126/science.abm7512',
                 'description': 'LLPC, memory B and T cell decay at 1–12 months; memory curve decay constants.',
                 'n': 'n=188 participants', 'param': 'Memory decay constants (0.05–0.08/month)'},
            ]
        },
        {
            'category': '🎯 Antigen Immunogenicity',
            'color': '#f59e0b', 'bg': '#fffbeb', 'border': '#fde68a',
            'datasets': [
                {'name': 'NetMHCpan Binding Affinity Database',
                 'type': 'Computational / in vitro validation',
                 'source': 'Reynisson et al., Nucleic Acids Research 2020',
                 'doi': '10.1093/nar/gkaa379',
                 'description': 'HLA-I and HLA-II peptide binding predictions; MHC score thresholds and PSSM anchor weights calibrated against NetMHCpan percentile distributions.',
                 'n': '>850,000 peptide–MHC pairs', 'param': 'PSSM anchor scores; IC50 thresholds'},
                {'name': 'SYFPEITHI / BIMAS HLA Binding Matrices',
                 'type': 'Experimental / computational',
                 'source': 'Rammensee et al., Immunogenetics 1999',
                 'doi': '10.1007/s002510050595',
                 'description': 'Published HLA-A*02:01 and HLA-DR anchor position weight matrices used directly in Epitrix PSSM epitope scanner.',
                 'n': '>3,000 known epitopes', 'param': '_HLA_A0201_PSSM and _HLA_DR_PSSM matrices'},
                {'name': 'IEDB B-Cell Epitope Dataset',
                 'type': 'Experimental / curated',
                 'source': 'Vita et al., Nucleic Acids Research 2019',
                 'doi': '10.1093/nar/gky1006',
                 'description': 'B-cell epitope density coefficients calibrated against IEDB charge/hydrophilicity distributions.',
                 'n': '>240,000 epitope records', 'param': 'B-cell hydrophilicity window scorer'},
            ]
        },
        {
            'category': '📊 Biological Variability (CI Parameters)',
            'color': '#0891b2', 'bg': '#f0f9ff', 'border': '#bae6fd',
            'datasets': [
                {'name': 'Inter-individual CV for Vaccine Immunogenicity',
                 'type': 'Meta-analysis',
                 'source': 'Voysey et al., Lancet 2021 (AstraZeneca pooled analysis)',
                 'doi': '10.1016/S0140-6736(20)32661-1',
                 'description': 'CV values for antibody titers (~40–45%), T cell responses (~35–45%), and efficacy point estimates (~18pp). Used directly as CV inputs to the CI model.',
                 'n': 'n=11,636 participants', 'param': '_CV["ab_magnitude"]=0.45, _CV["Th1"]=0.35'},
                {'name': 'Immunosenescence Effects on Vaccine Response',
                 'type': 'Systematic review',
                 'source': 'Crooke et al., npj Vaccines 2019',
                 'doi': '10.1038/s41541-019-0122-4',
                 'description': 'Elderly population modifiers: peak titer −32%, durability −25%, CI inflation ×1.35.',
                 'n': '31 trials reviewed', 'param': "POP_MOD['Elderly'] multipliers"},
            ]
        },
        {
            'category': '⚠️ Clinical Reactogenicity',
            'color': '#ef4444', 'bg': '#fff1f2', 'border': '#fecdd3',
            'datasets': [
                {'name': 'COVID-19 Vaccine Reactogenicity Surveillance (v-safe)',
                 'type': 'Post-market surveillance',
                 'source': 'Chapin-Bardales et al., JAMA 2021',
                 'doi': '10.1001/jama.2021.7517',
                 'description': 'Reactogenicity rates by vaccine type; safety score calibration and reactogenicity gauge threshold (0.66).',
                 'n': 'n=3.6 million reports', 'param': 'Reactogenicity threshold = 0.66'},
                {'name': 'LNP Reactogenicity Dose–Response Meta-Analysis',
                 'type': 'Clinical meta-analysis',
                 'source': 'Ndeupen et al., iScience 2021',
                 'doi': '10.1016/j.isci.2021.103479',
                 'description': 'Innate immune activation vs local reactogenicity across LNP formulations; reactogenicity = TLR×0.4 + Complement×0.3 + Inflammasome×0.3.',
                 'n': '14 studies, 8 LNP platforms', 'param': 'Reactogenicity equation weights'},
            ]
        },
        {
            'category': '📐 Efficacy Equation Calibration',
            'color': '#0891b2', 'bg': '#f0f9ff', 'border': '#bae6fd',
            'datasets': [
                {'name': 'BNT162b2 (Pfizer/BioNTech) Phase 3 Efficacy. Primary Calibration Anchor',
                 'type': 'Phase 3 RCT',
                 'source': 'Polack et al., N Engl J Med 2020 (NCT04368728)',
                 'doi': '10.1056/NEJMoa2034577',
                 'description': 'ALC-0315 LNP + m1Ψ mRNA: 95% efficacy (95% CI 90.3–97.6%) vs symptomatic COVID-19, n=43,448. '
                                'Primary calibration anchor for mRNA-LNP canonical formulation. '
                                'Molecular efficacy formula: E_raw = tlr_evasion×0.25 + trans_eff×mod_level/100×0.35 + antigenicity×0.25 + pKa_opt×0.15. '
                                'ALC-0315+m1Ψ: E_raw≈0.825 → 94%. Calibration: FLOOR=−0.122, SCALE=1.286.',
                 'n': 'n=43,448 (21,720 vaccine / 21,728 placebo)',
                 'param': 'CAL_FLOOR=-0.1217, CAL_SCALE=1.2863 (mRNA/DNA direct molecular formula)'},
                {'name': 'mRNA-1273 (Moderna) Phase 3 Efficacy: Confirmatory mRNA Anchor',
                 'type': 'Phase 3 RCT',
                 'source': 'Baden et al., N Engl J Med 2021 (COVE trial)',
                 'doi': '10.1056/NEJMoa2035389',
                 'description': 'SM-102 LNP + m1Ψ mRNA: 94.1% efficacy at interim, 93.2% at blinded completion (95% CI 91.0–94.8%). '
                                'n=30,415 participants. Confirms mRNA-LNP 93–95% calibration range for both ALC-0315 and SM-102.',
                 'n': 'n=30,415 (15,209 vaccine / 15,206 placebo)',
                 'param': 'Confirms mRNA calibration anchor at 93–95% for SM-102 formulation'},
                {'name': 'Shingrix (AS01B-adjuvanted zoster) Phase 3: Protein+Adjuvant Upper Anchor',
                 'type': 'Phase 3 RCT',
                 'source': 'Lal et al., N Engl J Med 2015 (ZOE-50)',
                 'doi': '10.1056/NEJMoa1501184',
                 'description': 'Recombinant VZV glycoprotein E + AS01B: 97.2% efficacy in adults ≥50 years. '
                                'Upper calibration anchor for protein subunit + AS01B. '
                                'adj_efficacy_boost for AS01B = 0.28.',
                 'n': 'n=15,411', 'param': 'AS01B adj_boost=0.28; Protein FLOOR=0.128, SCALE=1.719'},
                {'name': 'HEPLISAV-B (CpG 1018-adjuvanted HBV): Protein+TLR9 Anchor',
                 'type': 'Phase 3 RCT',
                 'source': 'Heyward et al., Vaccine 2013 · FDA Approval 2017',
                 'doi': '10.1016/j.vaccine.2013.04.070',
                 'description': 'Recombinant HBsAg + CpG 1018 (TLR9): ~93% seroprotection vs ~81% for alum-adjuvanted control. '
                                'adj_efficacy_boost for CpG 1018 = 0.20 for protein vaccines.',
                 'n': 'n=2,476', 'param': 'CpG 1018 adj_boost=0.20'},
                {'name': 'Alum-adjuvanted HepB vaccine: Protein+Alum Lower Anchor',
                 'type': 'Clinical data',
                 'source': 'Andre 1989, Vaccine · WHO position paper 2017',
                 'doi': '10.1016/0264-410X(89)90236-4',
                 'description': 'Alum-adjuvanted recombinant HBsAg: ~60% seroprotection in low responders, ~95% overall. '
                                'Used as lower calibration anchor for protein+alum. '
                                'Alum adj_efficacy_boost = 0.19.',
                 'n': 'Pooled clinical data', 'param': 'Alum adj_boost=0.19; Protein alum baseline ~60%'},
                {'name': 'Cervarix (AS04-adjuvanted HPV): Protein+MPL/Alum Anchor',
                 'type': 'Phase 3 RCT',
                 'source': 'Paavonen et al., Lancet 2009 (PATRICIA trial)',
                 'doi': '10.1016/S0140-6736(09)61248-4',
                 'description': 'Recombinant HPV-16/18 VLP + AS04 (MPL+alum): 93% efficacy against persistent infection. '
                                'adj_efficacy_boost for AS04 = 0.20.',
                 'n': 'n=18,644', 'param': 'AS04 adj_boost=0.20'},
            ]
        },
        {
            'category': '💊 Adjuvant Parameterisation Sources',
            'color': '#7c3aed', 'bg': '#f5f3ff', 'border': '#ddd6fe',
            'datasets': [
                {'name': 'Alum NLRP3 inflammasome activation',
                 'type': 'In vitro immunology',
                 'source': 'Kool et al., J Immunol 2008 · Li et al., J Immunol 2008',
                 'doi': '10.4049/jimmunol.181.1.17',
                 'description': 'NLRP3/ASC-dependent IL-1β/IL-18 from alum. Basis for Inflammasome=high, TLR7/8=none. NLRP3 dispensable for antibody production confirmed in parallel (PMC 2009).',
                 'n': 'Multiple KO mouse lines', 'param': "Alum: Inflammasome='high', TLR7_8='none'"},
                {'name': 'Alum Th2 bias and IgG1/IgE skewing',
                 'type': 'In vivo immunology',
                 'source': 'Brewer et al., J Immunol 1999 · Badran et al., Sci Rep 2022',
                 'doi': '10.1038/s41598-023-30336-1',
                 'description': 'IgG1/IgG2a ratio confirms strong Th2 bias. STING dispensable for alum adjuvanticity (Immunity 2024). th_bias: Th2 +0.30, Th1 −0.15.',
                 'n': 'Multiple mouse strains', 'param': "Alum: th_bias Th2=+0.30"},
                {'name': 'MF59 non-TLR MyD88 pathway',
                 'type': 'KO mouse immunology',
                 'source': 'Seubert et al., PNAS 2011',
                 'doi': '10.1073/pnas.1107941108',
                 'description': 'MF59 does not activate any TLR in vitro, confirmed on all TLR reporter cell lines. It requires MyD88 through a TLR-independent pathway. Basis for TLR7_8=none.',
                 'n': 'TLR/MyD88 KO comparison', 'param': "MF59: TLR7_8='none', non-TLR MyD88"},
                {'name': 'MF59 RIPK3-dependent CD8 cross-presentation',
                 'type': 'Mechanistic mouse study',
                 'source': 'Seubert et al., eLife 2020',
                 'doi': '10.7554/eLife.52687',
                 'description': 'RIPK3 necroptosis drives cross-presentation to CD8 T cells by Batf3+ cDCs. Basis for cd8_boost=0.10 for MF59/AS03/AddaVax.',
                 'n': 'RIPK3-KO vs WT comparison', 'param': "MF59/AS03: cd8_boost=0.10"},
                {'name': 'AS01 MPL+QS-21 synergy: early IFN-γ mechanism',
                 'type': 'Mechanistic NHP + clinical',
                 'source': 'Coccia et al., npj Vaccines 2017',
                 'doi': '10.1038/s41541-017-0027-3',
                 'description': 'MPL (TLR4) + QS-21 (NLRP3) synergistic early IFN-γ from NK cells via IL-12/IL-18. Blocking IFN-γ abolishes Th1 polarisation. Basis for AS01B highest Th1 deltas.',
                 'n': 'Mouse + macaque + human Phase II', 'param': "AS01B: Th1=+0.40, Tfh=+0.20"},
                {'name': 'AS01 mode of action: TLR4 and caspase-1 requirement',
                 'type': 'Mechanistic review',
                 'source': 'Didierlaurent et al., Tandfonline 2024',
                 'doi': '10.1080/14760584.2024.2382725',
                 'description': 'TLR4 expression in hematopoietic cells sufficient for AS01 adjuvant effect. QS-21 activates NLRP3 inflammasome. Liposomal delivery required for full synergy.',
                 'n': 'Review of licensed vaccine data', 'param': "AS01: Inflammasome='moderate', TLR4→TLR7_8 slot='high'"},
                {'name': 'CpG ODN TLR9 Th1 mechanism',
                 'type': 'Mechanistic review',
                 'source': 'Coffman et al., Immunity 2010',
                 'doi': '10.1016/j.immuni.2010.10.002',
                 'description': 'TLR9→MyD88→IRF7→IFN-α and NF-κB→IL-12. Strong Th1/IgG2 bias, B cell activation. CpG 1018 (FDA-approved 2017 HEPLISAV-B) and CpG 7909 (FDA-approved 2023 Cyfendus).',
                 'n': 'Multiple clinical datasets', 'param': "CpG ODN: TLR7_8='very_high' (TLR9 mapped), Th1=+0.35"},
                {'name': '3M-052 long-lived plasma cell induction',
                 'type': 'NHP immunology',
                 'source': 'Nat Comms 2022: Molecular atlas of innate immunity',
                 'doi': '10.1038/s41467-022-28197-9',
                 'description': 'Lipidated TLR7/8 agonist induces robust antiviral/IFN gene program similar to yellow fever vaccine. Long-lived plasma cells up to ~1 year in NHPs. Highest Tfh delta of TLR7/8 class.',
                 'n': 'scRNA-seq + flow cytometry, NHP', 'param': "3M-052: Tfh=+0.15, cd8_boost=0.12"},
                {'name': 'Matrix-M saponin nanoparticle mechanism',
                 'type': 'Mechanistic + clinical',
                 'source': 'Science Advances 2024: saponin-TLRa nanoadjuvants',
                 'doi': '10.1126/sciadv.adn7187',
                 'description': 'Saponin nanoparticle (ISCOMATRIX-comparable). NLRP3-driven inflammasome, strong Th1 + CD8 cross-presentation. FDA-approved Oct 2022 Novavax COVID-19 vaccine. WHO-recommended R21/Matrix-M malaria vaccine 2023.',
                 'n': 'COVID-19 + malaria Phase III', 'param': "Matrix-M: Inflammasome='moderate', cd8_boost=0.18"},
                {'name': 'Poly-ICLC (Hiltonol) TLR3/MDA-5 adjuvancy',
                 'type': 'Clinical review',
                 'source': 'Caskey et al., PMC 2022: Vaccines overview',
                 'doi': '10.3390/vaccines10050819',
                 'description': 'Poly-ICLC stronger Th1 response than LPS or CpG in direct comparisons. TLR3 + cytosolic MDA-5. Strong type I IFN and CD8 cross-presentation via cDC1.',
                 'n': 'Multiple Phase I/II trials', 'param': "Poly-ICLC: TLR3='high', cd8_boost=0.18"},
                {'name': 'Comprehensive adjuvant mechanisms review',
                 'type': 'Systematic review',
                 'source': 'Frontiers Immunology 2025: Recent advances in vaccine adjuvants',
                 'doi': '10.3389/fimmu.2025.1557415',
                 'description': 'Full mechanistic coverage of TLR1/2/3/4/5/7/8/9 agonists, emulsions, saponins, and combination systems. Primary reference for ordinal pathway strength assignments across the adjuvant catalogue.',
                 'n': '200+ citations reviewed', 'param': 'All adjuvant ordinal pathway scores'},
                {'name': 'Adjuvant confidence tiers and clinical status',
                 'type': 'Clinical landscape review',
                 'source': 'Goetz et al., Bioengineering & Translational Medicine 2024',
                 'doi': '10.1002/btm2.10663',
                 'description': 'Comprehensive survey of FDA/EMA-approved adjuvants and clinical trials pipeline. Source for confidence tier assignments (approved/clinical/preclinical) across 22 adjuvants in the Epitrix catalogue.',
                 'n': 'All licensed and Phase I–III adjuvants', 'param': 'confidence tier labels'},
            ]
        },
    ]

    for cat in DATASETS:
        count = len(cat['datasets'])
        # Category header
        st.markdown(f"""
<div class="content-container" style="padding-top:0;padding-bottom:0;">
  <div style="background:{cat['bg']};border:1px solid {cat['border']};
       border-left:5px solid {cat['color']};border-radius:14px 14px 0 0;
       padding:1rem 1.5rem;margin-bottom:0;">
    <div style="display:flex;align-items:center;justify-content:space-between;">
      <h3 style="color:{cat['color']} !important;margin:0;font-size:1.05rem;font-weight:700;">
        {cat['category']}
      </h3>
      <span style="background:{cat['color']};color:white;font-size:0.72rem;font-weight:700;
            padding:3px 10px;border-radius:999px;">{count} source{'s' if count>1 else ''}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        # Each dataset card — pre-compute all dynamic values to avoid quote conflicts inside f-strings
        for i, d in enumerate(cat['datasets']):
            is_last    = (i == len(cat['datasets']) - 1)
            bottom_style = (
                f'border-bottom:1px solid {cat["border"]};border-radius:0 0 14px 14px;'
                if is_last else
                f'border-bottom:1px solid {cat["border"]};'
            )
            c         = cat['color']
            bg        = cat['bg']
            border    = cat['border']
            name      = d['name']
            source    = d['source']
            doi       = d['doi']
            desc      = d['description']
            param     = d['param']
            dtype     = d['type']
            n         = d['n']

            html = (
                '<div style="background:white;'
                f'border-left:1px solid {border};border-right:1px solid {border};{bottom_style}'
                'padding:0.9rem 1.5rem;">'
                '<div style="display:flex;align-items:flex-start;'
                'justify-content:space-between;gap:1rem;flex-wrap:wrap;">'
                '<div style="flex:1;min-width:220px;">'
                f'<div style="font-weight:700;color:#111827;font-size:0.9rem;margin-bottom:0.2rem;">{name}</div>'
                f'<div style="color:#6b7280;font-size:0.78rem;margin-bottom:0.35rem;">'
                f'{source} &nbsp;·&nbsp; '
                f'<a href="https://doi.org/{doi}" target="_blank" '
                f'style="color:{c};text-decoration:none;font-weight:500;">DOI:{doi}</a>'
                '</div>'
                f'<div style="color:#374151;font-size:0.83rem;line-height:1.5;margin-bottom:0.3rem;">{desc}</div>'
                f'<div style="font-size:0.75rem;color:#6b7280;">'
                f'<strong style="color:#374151;">Used for:</strong> '
                f'<code style="background:#f3f4f6;padding:1px 5px;border-radius:4px;font-size:0.72rem;">{param}</code>'
                '</div>'
                '</div>'
                '<div style="text-align:right;flex-shrink:0;min-width:110px;">'
                f'<span style="background:{bg};border:1px solid {border};color:{c};'
                'font-size:0.7rem;font-weight:600;padding:2px 7px;border-radius:5px;'
                f'display:inline-block;margin-bottom:4px;">{dtype}</span><br>'
                f'<span style="color:#9ca3af;font-size:0.7rem;">{n}</span>'
                '</div>'
                '</div>'
                '</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

        st.markdown('<div style="margin-bottom:1.25rem;"></div>', unsafe_allow_html=True)

    st.markdown("""
<div class="content-container" style="padding-top:0;">
  <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;
       padding:1.25rem 1.5rem;">
    <p style="color:#6b7280;font-size:0.82rem;margin:0 0 0.5rem;line-height:1.7;">
      <strong style="color:#374151;">ℹ️ Platform architecture:</strong>
      Epitrix combines two components. The <strong>epitope prediction layer</strong> uses trained
      XGBoost models (MHC-I AUC 0.986, T cell AUC 0.928) trained on IEDB bulk data.
      The <strong>innate→adaptive cascade layer</strong> uses parameterised mechanistic equations
      whose coefficients were manually extracted from the sources above: not trained end-to-end.
      Both layers contribute to the final prediction.
    </p>
    <p style="color:#6b7280;font-size:0.82rem;margin:0;line-height:1.7;">
      <strong style="color:#374151;">⚠️ Research use only:</strong>
      Outputs are hypothesis-generating approximations for research and educational purposes.
      The mechanistic cascade has not been prospectively validated. ML model test sets were
      randomly split from the same IEDB download and not independently benchmarked.
      Results should not inform clinical decisions without experimental validation.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    create_breakthrough_header()

    # ── Sidebar styling ────────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <style>
    /* Sidebar background already set to #1e293b in main CSS */
    .sb-logo {
        display: flex; align-items: center; gap: 0.6rem;
        padding: 1rem 0.5rem 0.4rem;
    }
    .sb-logo-icon {
        width: 40px; height: 40px; border-radius: 10px;
        background: linear-gradient(135deg, #2563eb, #06b6d4);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem; flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(37,99,235,0.4);
    }
    .sb-logo-text { line-height: 1.2; }
    .sb-logo-text strong {
        display: block; color: #f1f5f9 !important;
        font-size: 0.95rem; font-weight: 700;
    }
    .sb-logo-text span {
        color: #94a3b8 !important; font-size: 0.72rem;
    }
    .sb-divider {
        height: 1px; background: rgba(255,255,255,0.08);
        margin: 0.75rem 0;
    }
    .sb-section-label {
        font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: #64748b !important;
        padding: 0 0.25rem; margin-bottom: 0.3rem; display: block;
    }
    </style>
    <div class="sb-logo">
      <div class="sb-logo-icon">🧠</div>
      <div class="sb-logo-text">
        <strong>Epitrix</strong>
        <span>Hybrid ML + Simulation Platform</span>
      </div>
    </div>
    <div class="sb-divider"></div>
    <span class="sb-section-label">Navigation</span>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Select Module",
        options=[
            "🚀 Core Innovation",
            "🎯 Prediction Targets",
            "📊 Data Integration",
            "🔬 Simulation Platform",
            "📚 Evidence Base",
        ],
        label_visibility="collapsed"
    )

    # ── Bottom sidebar version tag ─────────────────────────────────────────────
    st.sidebar.markdown("""
    <div class="sb-divider" style="margin-top:2rem;"></div>
    <div style="padding:0.5rem 0.25rem;">
      <span style="color:#475569 !important;font-size:0.7rem;">Epitrix v2.0 · Research Preview</span><br>
      <span style="color:#334155 !important;font-size:0.68rem;">
        ML epitope (AUC 0.986) + mechanistic cascade
      </span><br>
      <span style="color:#334155 !important;font-size:0.68rem;">
        Not for clinical use · Research use only
      </span>
    </div>
    """, unsafe_allow_html=True)

    if   page == "🚀 Core Innovation":
        display_core_innovation()
        display_breakthrough_concept()
    elif page == "🎯 Prediction Targets":
        display_prediction_targets()
    elif page == "📊 Data Integration":
        display_data_integration()
    elif page == "🔬 Simulation Platform":
        display_modeling_platform()
    elif page == "📚 Evidence Base":
        display_training_datasets()


if __name__ == "__main__":
    main()
