
# TRUTHLENS 
# This file contains the Streamlit UI + existing project flow.
# IMPORTANT:
#   - Do NOT change model loading, predict_text(), fetch_live_news(),
#     selected_source_ids, or prediction calculations if you only want UI.
#   - Do NOT rename model/data variables unless you also understand
#     every place where they are used.
#   - Existing ML/data logic is intentionally kept unchanged.

# 1. IMPORTS / LIBRARIES
import hashlib
import pandas as pd
import streamlit as st
from src.detector import load_or_train_model, predict_text
from src.news_scraper import available_source_labels, fetch_live_news
st.set_page_config(
    page_title="TruthLens — Fake News Detection",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# TRUTHLENS - COMPLETE UI CSS

st.markdown("""
<style>

/*
   GLOBAL APP 
*/

.stApp {
    background:
        radial-gradient(
            circle at 80% 0%,
            #0d2d4b 0%,
            #071727 45%,
            #030b13 100%
        );
    color: #eef6ff;
}


/* 
   MAIN CONTENT
*/

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    color: #f4f9ff !important;
}


/*
   SIDEBAR
*/

/* ---------- container ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06182a 0%, #071d32 55%, #092743 100%) !important;
    border-right: 1px solid #1d4d70 !important;
}

[data-testid="stSidebarContent"] {
    padding: 0 16px !important;              /* same left/right margin everywhere */
    scrollbar-gutter: auto !important;       /* removes Streamlit's extra side gap */
    scrollbar-width: thin;
    scrollbar-color: #1d4d70 transparent;
}
[data-testid="stSidebarHeader"]      { height: 2.75rem !important; padding: 0 !important; }
[data-testid="stSidebarUserContent"] { padding: 0 0 24px !important; }

/* ONE spacing rule for everything in the sidebar
   (Streamlit's default is 16px between every widget = random-looking gaps) */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 4px !important; }

/* ---------- brand ---------- */
.tl-brand      { display: flex; align-items: center; gap: 12px; }
.tl-logo       { flex: 0 0 44px; width: 44px; height: 44px; border-radius: 12px;
                 display: flex; align-items: center; justify-content: center;
                 background: linear-gradient(135deg, #12a8ff, #2463ff);
                 box-shadow: 0 0 18px rgba(36,99,255,.35); }
.tl-logo img   { width: 24px; height: 24px; }
.tl-brand-name { font-size: 25px; font-weight: 800; line-height: 1.05; letter-spacing: -.5px; color: #fff; }
.tl-brand-name span { color: #29a2ff; }
.tl-brand-sub  { margin-top: 4px; font-size: 11.5px; color: #9db7d0; }
.tl-tagline    { margin: 14px 0 8px; padding-bottom: 14px;
                 border-bottom: 1px solid rgba(80,150,230,.22);
                 font-size: 12px; font-weight: 700; color: #2da8ff; }

/* ---------- navigation buttons (all sidebar buttons) ---------- */
/* 1. Five Navigation Buttons - Refresh button  */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #0e83ff 0%, #1c54da 100%) !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 9px !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(14, 131, 255, 0.4) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 9px !important;
        padding: 0.5rem 0.85rem !important;
        color: #cbd5e1 !important;
        margin-bottom: 4px !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(14, 131, 255, 0.15) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
        color: #ffffff !important;
    }

    /* 2. Checkboxes  */
    section[data-testid="stSidebar"] input[type="checkbox"]:checked + div {
        background-color: #0284c7 !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 10px rgba(14, 131, 255, 0.5) !important;
    }
    section[data-testid="stSidebar"] input[type="checkbox"]:checked + div svg {
        fill: #ffffff !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stCheckbox"] {
        margin-top: 4px !important;
        margin-bottom: 4px !important;
        padding-top:1px !important;
        padding-bottom:1px !important;
    }

    /* 3. Slider */
    section[data-testid="stSidebar"] div[data-baseweb="slider"] div[role="progressbar"] {
        background: linear-gradient(90deg, #0284c7, #38bdf8) !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] {
        background-color: #0e83ff !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 10px rgba(14, 131, 255, 0.7) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }
/* ---------- refresh button ---------- */
[data-testid="stSidebar"] .st-key-refresh_news { margin-top: 12px; }
[data-testid="stSidebar"] .st-key-refresh_news .stButton button {
    justify-content: center !important;
    background: linear-gradient(90deg, #0fb7ef, #2389ff) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 6px 16px rgba(18,130,255,.25) !important;
}
[data-testid="stSidebar"] .st-key-refresh_news .stButton button * { justify-content: center !important; text-align: center !important; }
[data-testid="stSidebar"] .st-key-refresh_news .stButton button p { font-weight: 700 !important; }
[data-testid="stSidebar"] .st-key-refresh_news .stButton button:hover { filter: brightness(1.1); }

/* ---------- model panel ---------- */
.tl-model      { padding: 12px 14px; border: 1px solid #1d4d70; border-radius: 12px; background: rgba(8,32,52,.75); }
.tl-model-head { display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 700; color: #eef6ff; }
.tl-dot        { width: 8px; height: 8px; border-radius: 50%; background: #2ecc71; box-shadow: 0 0 0 3px rgba(46,204,113,.18); }
.tl-model-src  { margin-left: auto; font-size: 11.5px; font-weight: 500; color: #8ba8c8; }
.tl-stats      { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; padding-top: 12px;
                 border-top: 1px solid rgba(80,150,230,.18); }
.tl-stat-label { font-size: 11.5px; color: #8ba8c8; }
.tl-stat-value { margin-top: 2px; font-size: 20px; font-weight: 750; line-height: 1.1; color: #f4f9ff; }

/* ---------- quote + version ---------- */
.tl-quote   { margin-top: 22px; padding-left: 14px; border-left: 2px solid #299fff;
              font-size: 12px; font-style: italic; line-height: 1.5; color: #b8cee0; }
.tl-version { margin-top: 12px; padding-left: 14px; font-size: 10.5px; color: #6e91aa; }


/*
   MAIN PAGE TITLE
*/

.tl-title {
    font-size: clamp(34px, 4.2vw, 56px);

    font-weight: 850;

    letter-spacing: -2px;

    line-height: 1.05;

    margin: 7px 0;
}

.tl-title span {
    color: #299fff;
}

.tl-subtitle {
    color: #91afc8;

    font-size: 14px;

    margin-bottom: 18px;
}


/*
   METRICS
*/

div[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            #0b2339,
            #081a2c
        );

    border: 1px solid #1b4b70;

    border-radius: 14px;

    padding: 14px 17px;
}

div[data-testid="stMetricLabel"] {
    color: #83b6dc !important;
}

div[data-testid="stMetricValue"] {
    color: #f5faff !important;
}


/* 
   DATAFRAME
*/

.stDataFrame {
    border: 1px solid #1b4566;

    border-radius: 12px;

    overflow: hidden;
}


/* 
   TEXT AREA
*/

textarea {
    background: #091d31 !important;

    color: #eaf5ff !important;

    border: 1px solid #28577d !important;

    border-radius: 12px !important;
}


/* 
   INFO / SUCCESS / WARNING BOXES
*/

[data-testid="stAlert"] {
    border-radius: 12px !important;
}


/* 
   MOBILE
*/

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .tl-title {
        font-size: 36px;
    }

    .tl-name {
        font-size: 21px !important;
    }

    .tl-sub {
        font-size: 9px !important;
    }

    div[data-testid="stSidebar"] .stButton > button {
        font-size: 15px !important;
    }
}
</style>
""", unsafe_allow_html=True)



# 2. MODEL LOADING
#    Do NOT change unless you know how the ML model works.

@st.cache_resource
def get_model():
    return load_or_train_model()


@st.cache_data(ttl=300)
def get_latest_news(source_ids: tuple[str, ...], limit_per_feed: int):
    return fetch_live_news(source_ids=source_ids, limit_per_feed=limit_per_feed)


# 3. PREDICTION / RESULT DISPLAY
#    --Prediction result text, confidence display,
#    and FAKE/REAL messages can be changed here.

def show_prediction(text: str) -> None:
    result = predict_text(model, text)
    st.metric("Prediction", result.label.title(), f"{result.confidence:.1%} confidence")
    st.progress(result.confidence)
    st.caption(f"Fake: {result.fake_probability:.1%} | Real: {result.real_probability:.1%}")

    if result.label == "FAKE":
        st.error("This looks suspicious. Please verify it with trusted sources.")
    else:
        st.success("This looks like real news based on the current model.")


# 4. LIVE NEWS → ML CLASSIFICATION
#  --- Table data/column names are prepared here.
#    Keep prediction logic unchanged if only UI is being edited.

def classify_news(news_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in news_df.itertuples(index=False):
        result = predict_text(model, item.text)
        rows.append(
            {
                "source": item.source,
                "title": item.title,
                "prediction": result.label,
                "mark": "FAKE" if result.label == "FAKE" else "REAL",
                "confidence": f"{result.confidence:.1%}",
                "fake_probability": f"{result.fake_probability:.1%}",
                "real_probability": f"{result.real_probability:.1%}",
                "published": item.published,
                "link": item.link,
            }
        )
    return pd.DataFrame(rows)


# 5. UNIQUE TEXT-AREA KEY
#    Technical helper. 

def stable_text_key(text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"live_news_text_{digest}"


# 6. MAIN PAGE TITLE / HEADER
#  ----Main heading and subtitle text.

st.markdown('<div class="tl-title">Fake News <span>Detection System</span></div>',unsafe_allow_html=True)
st.markdown('<div class="tl-subtitle">Real News. Smarter Decisions.</div>',unsafe_allow_html=True)

model, metrics, model_source = get_model()
source_labels = available_source_labels()


# 7. SIDEBAR UI
#    -----Logo, TruthLens name, menu labels, source
#    selector, slider, refresh button, model information, etc.


with st.sidebar:

    
    # TRUTHLENS BRAND

    st.markdown(
        """
        <style>
        .truthlens-brand {
            padding: 8px 4px 18px 4px;
            border-bottom: 1px solid rgba(80, 150, 230, 0.22);
            margin-bottom: 18px;
        }

        .truthlens-brand-name {
            font-size: 25px;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 5px;
        }

        .truthlens-blue {
            color: #2498ff;
        }

        .truthlens-white {
            color: #ffffff;
        }

        .truthlens-subtitle {
            color: #8ba8c8;
            font-size: 11px;
            margin-top: 4px;
        }

        .truthlens-tagline {
            color: #2da8ff;
            font-size: 12px;
            font-weight: 600;
            margin-top: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Logo + name using native Streamlit components

    with st.sidebar:
        c1, c2 = st.columns([1, 4])
        with c1:
            st.markdown(
                """
                <div style="
                    width: 44px;
                    height: 44px;
                    border-radius: 12px;
                    background: linear-gradient(135deg, #12a8ff, #0070f3);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 22px;
                    box-shadow: 0 0 16px rgba(18, 168, 255, 0.45);
                ">
                    🔍
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                """
                <div style="line-height: 1.1; margin-top: -2px;">
                    <div style="
                        font-size: 1.95rem; 
                        font-weight: 900; 
                        letter-spacing: -0.03em; 
                        color: #ffffff;
                    ">
                        Truth<span style="color: #38bdf8; text-shadow: 0 0 14px rgba(56, 189, 248, 0.6);">Lens</span>
                    </div>
                    <div style="font-size: 0.74rem; color: #8fa6c1; font-weight: 500; margin-top: 3px;">
                        Fake News Detection System
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div style="
                text-align: center; 
                font-size: 0.75rem; 
                color: #38bdf8; 
                font-weight: 600; 
                letter-spacing: 0.04em; 
                margin-top: 10px; 
                margin-bottom: 24px;
            ">
                Analyze · Verify · Stay Informed
            </div>
            """,
            unsafe_allow_html=True
        )

        # SIDEBAR NAVIGATION----
        # Dashboard button
        # Live News button
        # Check News button
        # etc.

    if "page" not in st.session_state:
            st.session_state.page = "Dashboard"

    if st.button("   Dashboard", key="nav_dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"
            st.rerun()

    if st.button("   Live News", key="nav_live", use_container_width=True):
            st.session_state.page = "Live News"
            st.rerun()

    if st.button("   Check News", key="nav_check", use_container_width=True):
            st.session_state.page = "Check News"
            st.rerun()

    if st.button("   Model Info", key="nav_model", use_container_width=True):
            st.session_state.page = "Model Info"
            st.rerun()

    if st.button("   About", key="nav_about", use_container_width=True):
            st.session_state.page = "About"
            st.rerun()

        # NEWS SOURCES 

    st.markdown(
            '<div class="tl-section" style="margin-top: 18px; margin-bottom: 18px; font-weight: 600; color: #cbd5e1;">News Sources</div>',
            unsafe_allow_html=True
        )


    # SOURCE CHECKBOXES
    # Existing source_labels is still being used.
    # No news source/data logic is changed.

    selected_source_ids_list = []

    for source_id, source_name in source_labels.items():

        # Default = selected
        is_selected = st.checkbox(
            source_name,
            value=True,
            key=f"source_checkbox_{source_id}"
        )

        if is_selected:
            selected_source_ids_list.append(source_id)

    selected_source_ids = tuple(selected_source_ids_list)


    # ITEMS PER SOURCE

    st.markdown(
        '<div class="tl-section tl-items-title">Items per source</div>',
        unsafe_allow_html=True
    )

    limit_per_feed = st.select_slider(
    "Items per source",
    options=[3, 5, 8, 10, 15, 20],
    value=20,
    label_visibility="collapsed"
    
)

    st.markdown(
        f"""
        <div style="
            text-align:center;
            margin-top:-4px;
            margin-bottom:20px;
            color:#9fc9ff;
            font-size:12px;
            font-weight:550;
        ">
            Showing <span style="color:#ffffff;">{limit_per_feed}</span> items per source
        </div>
        """,
        unsafe_allow_html=True
    )

    # REFRESH NEWS BUTTON

    if st.button(
        "🔄  Refresh News",
        type="primary",
        use_container_width=True,
        key="btn_refresh_news"
    ):
        get_latest_news.clear()
        st.rerun()

    # SEPARATOR

    st.markdown(
        '<div class="tl-divider"></div>',
        unsafe_allow_html=True
    )

    # MODEL 
    # TRAINING ROWS
    

    st.metric(
        "Training rows",
        int(metrics.get("total_rows", 0))
    )

    st.metric(
        "Accuracy",
        f"{metrics.get('accuracy', 0):.1%}"
    )

    st.metric(
        "Model",
        str(model_source)
    )


    # SIDEBAR QUOTE

    st.markdown("""
    <div class="tl-quote">
        “A more informed<br>
        world is a safer world.”
    </div>

    <div class="tl-small-line"></div>
    """, unsafe_allow_html=True)

    # VERSION

    st.markdown("""
    <div class="tl-version">
        Version 1.0
    </div>
    """, unsafe_allow_html=True)

# 8. MODEL INFO / ABOUT PAGES

if st.session_state.page == "Model Info":

    st.markdown("##  Model Information")

    st.markdown("""
    ### Fake News Detection Model

    This project uses a Machine Learning model to classify
    news text as **REAL** or **FAKE**.

    **Model:** Logistic Regression

    **Text Processing:** TF-IDF Vectorization

    **Prediction Output:**
    - REAL — the model predicts the news as likely real
    - FAKE — the model predicts the news as likely fake

    **Evaluation Metrics:**
    - Accuracy
    - Precision
    - Recall
    - F1-Score

    The prediction is based on patterns learned from the
    training dataset. The result should be verified with
    reliable news sources.
    """)

    st.info("The model's prediction is an ML-based result and should not be treated as absolute fact.")

    st.stop()


if st.session_state.page == "About":

    st.markdown("##  About TruthLens")

    st.markdown("""
    ### TruthLens — Fake News Detection System

    TruthLens is a Machine Learning based application
    designed to analyze news text and classify it as
    **REAL** or **FAKE**.

    ### How it works

    **1. Enter News**
    
    The user provides a news headline or article text.

    **2. Text Processing**
    
    The text is converted into numerical features using
    the trained text vectorizer.

    **3. Machine Learning Prediction**
    
    The trained model analyzes the features and generates
    a prediction.

    **4. Result**
    
    The system displays the predicted category along with
    confidence information.

    ### Important Note

    No automated system can guarantee that every news
    article is true or false. Important information should
    always be verified using multiple reliable sources.
    """)

    st.stop()

# 9. MAIN TABS
#    Dashboard / Live News / Check Text tab names.
# ============================================================

tab_dashboard, tab_live, tab_manual = st.tabs(["Dashboard", "Live News", "Check Text"])

# --- DASHBOARD TAB

with tab_dashboard:
    if not selected_source_ids:
        st.warning("Select at least one source from the sidebar.")
    else:
        with st.spinner("Fetching and checking latest RSS headlines..."):
            dashboard_news, dashboard_errors = get_latest_news(
                selected_source_ids,
                limit_per_feed,
            )

        if dashboard_errors:
            with st.expander("Feed errors"):
                for error in dashboard_errors:
                    st.write(f"{error.source} / {error.feed}: {error.message}")

        if dashboard_news.empty:
            st.warning("No live news found. Try refreshing again.")
        else:
            dashboard_df = classify_news(dashboard_news)
            fake_count = int((dashboard_df["prediction"] == "FAKE").sum())
            real_count = int((dashboard_df["prediction"] == "REAL").sum())

            col_total, col_fake, col_real = st.columns(3)
            col_total.metric("Checked news", len(dashboard_df))
            col_fake.metric("Fake marked", fake_count)
            col_real.metric("Real marked", real_count)

            st.subheader("Fake and real marks")
            st.dataframe(
                dashboard_df,
                use_container_width=True,
                hide_index=True,
                column_config={"link": st.column_config.LinkColumn("link")},
            )

            fake_df = dashboard_df[dashboard_df["prediction"] == "FAKE"]
            if not fake_df.empty:
                st.subheader("Fake news marked by model")
                st.dataframe(
                    fake_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={"link": st.column_config.LinkColumn("link")},
                )
            else:
                st.info("No live headline is currently marked as FAKE by the model.")


# ---LIVE NEWS TAB

with tab_live:
    if not selected_source_ids:
        st.warning("Select at least one source from the sidebar.")
    else:
        with st.spinner("Fetching latest RSS headlines..."):
            live_news, errors = get_latest_news(selected_source_ids, limit_per_feed)

        if errors:
            with st.expander("Feed errors"):
                for error in errors:
                    st.write(f"{error.source} / {error.feed}: {error.message}")

        if live_news.empty:
            st.warning("No live news found. Try refreshing again.")
        else:
            st.metric("Live items", len(live_news))

            display_df = live_news[["source", "feed", "title", "published", "link"]]
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={"link": st.column_config.LinkColumn("link")},
            )

            options = [
                f"{row.source}: {row.title}"
                for row in live_news.itertuples(index=False)
            ]
            selected_option = st.selectbox("Choose a headline to check", options)
            selected_index = options.index(selected_option)
            selected_item = live_news.iloc[selected_index]

            st.subheader(selected_item["title"])
            if selected_item["summary"]:
                st.write(selected_item["summary"])
            if selected_item["link"]:
                st.link_button("Open original article", selected_item["link"])

            original_text = str(selected_item["text"])
            edited_text = st.text_area(
                "Selected RSS text",
                value=original_text,
                height=200,
                key=stable_text_key(original_text),
            )

            if st.button("Check selected headline", type="primary", use_container_width=True):
                if edited_text != original_text:
                    st.error("Error: this live news text was changed. Please refresh or use the original RSS text.")
                else:
                    show_prediction(original_text)

# --- CHECK TEXT / MANUAL NEWS TAB

with tab_manual:
    sample_text = (
        "Paste any news headline or short article here to check whether it looks "
        "fake or real."
    )
    user_text = st.text_area(
        "News text",
        value=sample_text,
        height=220,
        placeholder="Paste a headline or article summary...",
    )

    if st.button("Check news", type="primary", use_container_width=True):
        if user_text.strip():
            show_prediction(user_text)
        else:
            st.warning("Enter some news text first.")
