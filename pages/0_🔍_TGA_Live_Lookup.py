"""
pages/0_🔍_TGA_Live_Lookup.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Streamlit page: search training.gov.au directly and load a unit into the
rsd-convert pipeline without uploading a CSV.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="TGA Live Lookup", page_icon="🔍", layout="wide")

st.title("🔍 TGA Live Unit Lookup")
st.caption(
    "Search training.gov.au directly — no CSV needed. "
    "Select a unit to load it straight into the RSD generator."
)

# ---------------------------------------------------------------------------
# Lazy import so app still loads if zeep isn't installed
# ---------------------------------------------------------------------------
try:
    from core.tga_client import TGAClient
    _TGA_AVAILABLE = True
except ImportError:
    _TGA_AVAILABLE = False


if not _TGA_AVAILABLE:
    st.error(
        "**zeep** is not installed. Run `pip install zeep` then restart the app."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Credentials sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("TGA Credentials")
    tga_env = st.selectbox(
        "Environment",
        ["sandbox", "production"],
        index=0,
        help="Use sandbox for testing. Contact tgaproject@education.gov.au for production access.",
    )
    tga_user = st.text_input("Username", value="WebService.Read")
    tga_pass = st.text_input("Password", value="Asdf098", type="password")

@st.cache_resource(show_spinner=False)
def get_client(env: str, user: str, pwd: str) -> TGAClient:
    return TGAClient(username=user, password=pwd, env=env)

client = get_client(tga_env, tga_user, tga_pass)

# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------
with st.expander("🔌 Connection status", expanded=False):
    if st.button("Test connection"):
        with st.spinner("Pinging TGA…"):
            ok = client.ping()
        if ok:
            st.success("Connected to TGA successfully.")
        else:
            st.error(
                "Could not connect to TGA. Check credentials and network access. "
                "The sandbox environment is only reachable from public internet."
            )

# ---------------------------------------------------------------------------
# Search form
# ---------------------------------------------------------------------------
st.subheader("Search units of competency")

col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    search_title = st.text_input("Unit title contains…", placeholder="e.g. communication")
with col2:
    search_code = st.text_input("Unit code", placeholder="e.g. BSBCMM411")
with col3:
    search_tp = st.text_input("Training package", placeholder="e.g. BSB")

include_superseded = st.checkbox("Include superseded units", value=False)

if st.button("🔍 Search", type="primary"):
    if not search_title and not search_code and not search_tp:
        st.warning("Enter at least one search term.")
    else:
        with st.spinner("Searching TGA…"):
            try:
                results = client.search_units(
                    title=search_title,
                    code=search_code,
                    training_package_code=search_tp,
                    include_superseded=include_superseded,
                    page_size=100,
                )
                st.session_state["tga_search_results"] = results
            except Exception as exc:
                st.error(f"Search failed: {exc}")
                st.session_state["tga_search_results"] = []

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
results = st.session_state.get("tga_search_results", [])

if results:
    st.write(f"**{len(results)} unit(s) found**")
    df_results = pd.DataFrame(results)

    # Let user pick a row
    selected_code = st.selectbox(
        "Select unit to load",
        options=df_results["code"].tolist(),
        format_func=lambda c: f"{c} — {df_results.loc[df_results['code']==c, 'title'].iloc[0]}",
    )

    st.dataframe(
        df_results[["code", "title", "training_package_code", "status"]],
        use_container_width=True,
        hide_index=True,
    )

    if st.button(f"⬇️ Load {selected_code} into RSD generator", type="primary"):
        with st.spinner(f"Fetching full detail for {selected_code}…"):
            try:
                df_unit = client.unit_to_dataframe(selected_code)
                unit_detail = client.get_unit(selected_code)
                st.session_state["tga_loaded_unit_code"] = selected_code
                st.session_state["tga_loaded_unit_title"] = unit_detail.get("title", "")
                st.session_state["tga_loaded_df"] = df_unit
                st.success(
                    f"✅ Loaded **{selected_code}** — "
                    f"{len(df_unit)} performance criteria across "
                    f"{df_unit['element_num'].nunique()} elements."
                )
            except Exception as exc:
                st.error(f"Failed to load unit: {exc}")

# ---------------------------------------------------------------------------
# Preview loaded unit
# ---------------------------------------------------------------------------
if "tga_loaded_df" in st.session_state:
    df = st.session_state["tga_loaded_df"]
    code = st.session_state.get("tga_loaded_unit_code", "")
    title = st.session_state.get("tga_loaded_unit_title", "")

    st.divider()
    st.subheader(f"📋 Loaded: {code} — {title}")
    st.dataframe(df, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        csv = df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download as CSV",
            data=csv,
            file_name=f"{code}_elements_pcs.csv",
            mime="text/csv",
        )
    with col_b:
        if st.button("▶️ Send to RSD Generator (main page)"):
            # Store in session state keys that app.py reads from
            st.session_state["uploaded_df"] = df
            st.session_state["source_filename"] = f"{code}_tga_live.csv"
            st.switch_page("app.py")
