# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Argo Fleet Dashboard", layout="wide")

# --- Helper functions ---
@st.cache_data
def load_location_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

@st.cache_data
def load_profile_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ["lat", "lon", "pressure", "temperature", "salinity"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "rho" not in df.columns and {"temperature", "salinity", "pressure"}.issubset(df.columns):
        df["rho"] = 1000 + 0.8 * df["salinity"] - 0.2 * df["temperature"]
    return df

@st.cache_data
def load_bgc_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ["lat", "lon", "pressure", "temperature", "salinity", "doxy", "chla", "bbp700"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def make_section_chart(df, var, label):
    if "time" not in df.columns or "pressure" not in df.columns:
        return None
    fig = px.scatter(
        df, x="time", y="pressure", color=var,
        hover_data=["lat", "lon"],
        color_continuous_scale="Viridis",
        labels={var: label, "pressure": "Pressure (dbar)"},
        title=f"Section: {label}"
    )
    fig.update_yaxes(autorange="reversed")
    return fig

def make_overlaid_profiles(df, var, label):
    if "pressure" not in df.columns or var not in df.columns:
        return None
    fig = go.Figure()
    groups = df.groupby(df["time"].dt.strftime("%Y-%m-%d") if "time" in df.columns else np.arange(len(df)))
    for name, g in groups:
        fig.add_trace(go.Scatter(
            x=g[var], y=g["pressure"], mode="markers",
            marker=dict(size=6, color=g[var], colorscale="Viridis", showscale=True),
            name=str(name), hovertext=g["time"] if "time" in g else None
        ))
    fig.update_yaxes(autorange="reversed", title="Pressure (dbar)")
    fig.update_xaxes(title=label)
    fig.update_layout(title=f"Overlaid profiles: {label}")
    return fig

# --- File paths ---
LOC_PATH = "/home/td/Documents/argo/location.csv"
PROF_PATH = "/home/td/Documents/argo/merged_2025_cleaned.csv"
BGC_PATH = "/home/td/Documents/argo/cleaned.csv"

# Load data
try:
    loc_df = load_location_csv(LOC_PATH)
    prof_df = load_profile_csv(PROF_PATH)
    bgc_df = load_bgc_csv(BGC_PATH)
except Exception as e:
    st.error(f"Error loading CSVs: {e}")
    st.stop()

# Sidebar filters
st.sidebar.title("Filters")
float_col_name = "file" if "file" in loc_df.columns else loc_df.columns[0]
float_list = sorted(loc_df[float_col_name].dropna().unique().tolist())
selected_float = st.sidebar.selectbox("Select float (file)", options=["All"] + float_list)

# --- Date range filter ---
min_date = prof_df["time"].min() if "time" in prof_df.columns else None
max_date = prof_df["time"].max() if "time" in prof_df.columns else None
if min_date is not None and max_date is not None:
    default_start = max(min_date, max_date - pd.Timedelta(days=7))
    date_range = st.sidebar.date_input(
        "Profile date range",
        value=(default_start.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    else:
        start_date = pd.to_datetime(date_range)
        end_date = start_date + pd.Timedelta(days=1)
else:
    start_date = end_date = None

# --- Main layout ---
st.title("Argo Float Dashboard")

tabs = st.tabs([
    "Map & Metadata",
    "Profiles",
    "T/S Diagram",
    "Section Charts",
    "Overlaid Profiles",
    "Biogeochemical"
])

# --- Map & Metadata ---
with tabs[0]:
    st.header("Trajectory Map")
    if selected_float == "All":
        df_last = (
            loc_df.dropna(subset=["latitude", "longitude"])
            .sort_values("date")
            .groupby(float_col_name).last().reset_index()
        )
        map_df = df_last.rename(columns={"longitude": "lon", "latitude": "lat"})
    else:
        fmask = loc_df[float_col_name] == selected_float
        map_df = loc_df[fmask].dropna(subset=["latitude", "longitude"]).sort_values("date")
        map_df = map_df.rename(columns={"longitude": "lon", "latitude": "lat"})
    if map_df.empty:
        st.info("No location points available for mapping.")
    else:
        view = pdk.ViewState(
            latitude=map_df["lat"].mean(),
            longitude=map_df["lon"].mean(),
            zoom=4,
            pitch=0,
        )
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[lon, lat]',
            get_color='[0, 128, 255, 160]',
            get_radius=20000,
            pickable=True,
        )
        tooltip = {
            "html": "<b>Float:</b> {" + float_col_name + "}<br/>"
                    "<b>Date:</b> {date}<br/>"
                    "Lat: {lat} | Lon: {lon}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
        r = pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip,
                     map_style="mapbox://styles/mapbox/light-v9")
        st.pydeck_chart(r)
    st.subheader("Metadata")
    if selected_float == "All":
        st.dataframe(loc_df.head(20))
    else:
        st.dataframe(loc_df[loc_df[float_col_name] == selected_float])

# --- Profiles ---
with tabs[1]:
    st.header("Profiles")
    prof_filtered = prof_df.copy()
    if "file" in prof_filtered.columns and selected_float != "All":
        prof_filtered = prof_filtered[prof_filtered["file"] == selected_float]

    if start_date is not None and end_date is not None and "time" in prof_filtered.columns:
        prof_filtered = prof_filtered[
            (prof_filtered["time"] >= start_date) & (prof_filtered["time"] < end_date)
        ]

    if prof_filtered.empty:
        st.info("No profiles for filters")
    else:
        # Temperature
        if {"temperature", "pressure"} <= set(prof_filtered.columns):
            fig = px.scatter(
                prof_filtered,
                x="temperature", y="pressure",
                hover_data=["time", "lat", "lon"],
                title="Temperature profile",
                labels={"temperature": "T (°C)", "pressure": "Pressure (dbar)"},
                color="time",
                color_continuous_scale="Viridis"
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        # Salinity
        if {"salinity", "pressure"} <= set(prof_filtered.columns):
            fig = px.scatter(
                prof_filtered,
                x="salinity", y="pressure",
                hover_data=["time", "lat", "lon"],
                title="Salinity profile",
                labels={"salinity": "S (psu)", "pressure": "Pressure (dbar)"},
                color="time",
                color_continuous_scale="Viridis"
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        # Density
        if {"rho", "pressure"} <= set(prof_filtered.columns):
            fig = px.scatter(
                prof_filtered,
                x="rho", y="pressure",
                hover_data=["time", "lat", "lon"],
                title="Density profile",
                labels={"rho": "ρ (kg/m³)", "pressure": "Pressure (dbar)"},
                color="time",
                color_continuous_scale="Viridis"
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

# --- T/S Diagram ---
with tabs[2]:
    st.header("T–S Diagram")
    if {"temperature", "salinity"} <= set(prof_filtered.columns):
        fig = px.scatter(prof_filtered, x="temperature", y="salinity",
                         color="pressure" if "pressure" in prof_filtered.columns else None,
                         hover_data=["time", "lat", "lon"],
                         title="T–S Diagram")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No temperature/salinity columns available")

# --- Section Charts ---
with tabs[3]:
    st.header("Section Charts")
    for var, label in [
        ("temperature", "Temperature (°C)"),
        ("salinity", "Salinity (psu)"),
        ("rho", "Density (kg/m³)")
    ]:
        if var in prof_filtered.columns:
            fig = make_section_chart(prof_filtered, var, label)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

# --- Overlaid Profiles ---
with tabs[4]:
    st.header("Overlaid Profiles")
    for var, label in [
        ("temperature", "Temperature (°C)"),
        ("salinity", "Salinity (psu)"),
        ("rho", "Density (kg/m³)")
    ]:
        if var in prof_filtered.columns:
            fig = make_overlaid_profiles(prof_filtered, var, label)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

# --- Biogeochemical Argo ---
with tabs[5]:
    st.header("Biogeochemical Argo Dashboard")

    bgc_filtered = bgc_df.copy()
    if start_date is not None and end_date is not None and "time" in bgc_filtered.columns:
        bgc_filtered = bgc_filtered[
            (bgc_filtered["time"] >= start_date) & (bgc_filtered["time"] < end_date)
        ]

    if bgc_filtered.empty:
        st.info("No BGC profiles for selected filters")
    else:
        for var, label in [
            ("doxy", "Dissolved Oxygen (µmol/kg)"),
            ("chla", "Chlorophyll-a (mg/m³)"),
            ("bbp700", "Backscatter at 700nm (1/m)")
        ]:
            if {"pressure", var} <= set(bgc_filtered.columns):
                st.subheader(label)
                # Profile plot
                fig = px.scatter(bgc_filtered, x=var, y="pressure",
                                 color=var, color_continuous_scale="Viridis",
                                 hover_data=["time", "lat", "lon"],
                                 title=f"{label} Profile")
                fig.update_yaxes(autorange="reversed", title="Pressure (dbar)")
                st.plotly_chart(fig, use_container_width=True)

                # Section chart
                fig = make_section_chart(bgc_filtered, var, label)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Overlaid profiles
                fig = make_overlaid_profiles(bgc_filtered, var, label)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
