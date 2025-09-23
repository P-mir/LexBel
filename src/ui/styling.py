
"""The branding..."""
def get_custom_css() -> str:
    return """
    <style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    /* Root variables - LexBel Brand Colors */
    :root {
        --lexbel-primary: #1a365d;      /* Deep navy blue */
        --lexbel-secondary: #2c5282;    /* Royal blue */
        --lexbel-accent: #d4af37;       /* Legal gold */
        --lexbel-success: #2f855a;      /* Forest green */
        --lexbel-warning: #d97706;      /* Amber */
        --lexbel-danger: #c53030;       /* Deep red */
        --lexbel-light: #f7fafc;        /* Off white */
        --lexbel-dark: #1a202c;         /* Charcoal */
        --lexbel-gray: #718096;         /* Neutral gray */
        --lexbel-border: #e2e8f0;       /* Light border */
    }

    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--lexbel-dark);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .main {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: var(--lexbel-primary);
        font-weight: 700;
    }

    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.5px;
    }

    h2 {
        font-size: 1.875rem !important;
        color: var(--lexbel-secondary);
        margin-top: 2rem !important;
    }

    h3 {
        font-size: 1.5rem !important;
        color: var(--lexbel-secondary);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--lexbel-primary) 0%, var(--lexbel-secondary) 100%);
        padding-top: 2rem;
    }

    /* Sidebar text color - more selective */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span:not([data-testid="stMetricValue"]):not([data-testid="stMetricLabel"]) {
        color: white !important;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--lexbel-accent) !important;
        font-weight: 600;
    }

    /* Sidebar metrics - keep readable */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        border-radius: 8px;
    }

    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: var(--lexbel-accent) !important;
        font-size: 0.75rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar success/error messages */
    [data-testid="stSidebar"] .stSuccess,
    [data-testid="stSidebar"] .stError {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 0.5rem;
    }

    /* Logo container */
    .lexbel-logo-container {
        text-align: center;
        padding: 1rem 0 2rem 0;
        border-bottom: 2px solid var(--lexbel-accent);
        margin-bottom: 2rem;
    }

    .lexbel-logo {
        max-width: 180px;
        height: auto;
        filter: brightness(0) invert(1);
    }

    .lexbel-tagline {
        color: var(--lexbel-accent) !important;
        font-size: 0.875rem;
        font-style: italic;
        margin-top: 0.5rem;
        font-family: 'Playfair Display', serif;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--lexbel-primary) 0%, var(--lexbel-secondary) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(26, 54, 93, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(26, 54, 93, 0.3);
        background: linear-gradient(135deg, var(--lexbel-secondary) 0%, var(--lexbel-primary) 100%);
    }

    /* Metrics/KPI cards */
    [data-testid="stMetric"] {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--lexbel-accent);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    [data-testid="stMetric"] label {
        color: var(--lexbel-gray) !important;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--lexbel-primary) !important;
        font-size: 2rem !important;
        font-weight: 700;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Success messages */
    [data-baseweb="notification"] {
        border-radius: 8px;
        background: white;
        border-left: 4px solid var(--lexbel-success);
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 8px;
        border: 1px solid var(--lexbel-border);
        font-weight: 600;
        color: var(--lexbel-primary);
    }

    .streamlit-expanderHeader:hover {
        border-color: var(--lexbel-accent);
    }

    /* Text input */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid var(--lexbel-border);
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--lexbel-accent);
        box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.1);
    }

    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid var(--lexbel-border);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        color: var(--lexbel-gray);
        border: 2px solid var(--lexbel-border);
        border-bottom: none;
    }

    .stTabs [aria-selected="true"] {
        background: var(--lexbel-primary);
        color: white !important;
        border-color: var(--lexbel-primary);
    }

    /* Charts */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Data tables */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Chat message styling */
    .stChatMessage {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-left: 4px solid var(--lexbel-primary);
    }

    .stChatMessage[data-testid="assistant-message"] {
        border-left: 4px solid var(--lexbel-accent);
    }

    /* Loading spinner */
    .stSpinner > div {
        border-top-color: var(--lexbel-accent) !important;
    }

    /* Dashboard cards */
    .dashboard-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 3px solid var(--lexbel-accent);
        margin-bottom: 1rem;
    }

    .dashboard-card h3 {
        color: var(--lexbel-primary);
        margin-top: 0;
        font-size: 1.25rem;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 600;
    }

    .status-success {
        background: #c6f6d5;
        color: #22543d;
    }

    .status-warning {
        background: #feebc8;
        color: #7c2d12;
    }

    /* Tooltips */
    [data-testid="stTooltipIcon"] {
        color: var(--lexbel-accent);
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        h1 {
            font-size: 2rem !important;
        }

        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--lexbel-light);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--lexbel-gray);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--lexbel-primary);
    }

    .lexbel-footer {
        text-align: center;
        padding: 2rem 0;
        color: var(--lexbel-gray);
        font-size: 0.875rem;
        border-top: 1px solid var(--lexbel-border);
        margin-top: 3rem;
    }

    a[href^="mailto:"] {
        transition: opacity 0.3s ease, color 0.3s ease;
    }

    a[href^="mailto:"]:hover {
        opacity: 1 !important;
        color: var(--lexbel-accent) !important;
        text-decoration: underline;
    }

    /* Sidebar email link */
    [data-testid="stSidebar"] a[href^="mailto:"] {
        color: white !important;
    }

    [data-testid="stSidebar"] a[href^="mailto:"]:hover {
        color: var(--lexbel-accent) !important;
    }
    </style>
    """

def get_lexbel_logo_html() -> str:
    """Return HTML for LexBel logo display."""
    return """
    <div class="lexbel-logo-container">
        <img src="app/static/logo" class="lexbel-logo" alt="LexBel Logo"/>
        <div class="lexbel-tagline">Intelligence Juridique Belge</div>
    </div>
    """
