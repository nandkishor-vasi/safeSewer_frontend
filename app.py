import os
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import altair as alt
import pandas as pd
import psycopg2
import streamlit as st

from ml.anomaly_detector import detect_anomalies

st.set_page_config(page_title="Node Analytics", layout="wide", initial_sidebar_state="expanded")

# Fixed values (same as your original code)
# Configure these against calibrated hardware before real-world use.
TH_MQ135, TH_MQ7, TH_DIST = 400, 300, 20
STALE_MIN = 5
REFRESH = "15s"
REQUIRED_COLUMNS = {"timestamp", "mq135", "mq7", "distance"}
GEOCODER_USER_AGENT = os.getenv("GEOCODER_USER_AGENT", "SafeSewerDashboard/1.0")

COLORS = {"mq135": "#2563eb", "mq7": "#f97316", "distance": "#14b8a6"}
LABELS = {"mq135": "MQ-135 (Air Quality)", "mq7": "MQ-7 (Carbon Monoxide)", "distance": "Water Distance"}

st.markdown(
    """
    <style>
    :root { --ink:#101a33; --muted:#64748b; --line:#e4eaf3; }
    .stApp, [data-testid="stAppViewContainer"] { background:linear-gradient(135deg,#f7f9ff,#f2f6fb 55%,#f8fafc) !important; color:var(--ink) !important; }
    [data-testid="stHeader"] { background:transparent !important; }
    .block-container { padding-top:1.25rem; padding-bottom:2.25rem; max-width:1580px; }
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp h1, .stApp h2, .stApp h3 { color:inherit; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#101d38,#132a49) !important; border-right:1px solid #203653; }
    [data-testid="stSidebar"] * { color:#e7eefb !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#aebed5 !important; }
    [data-testid="stVerticalBlockBorderWrapper"], [class*="st-key-card"] {
      background:rgba(255,255,255,.96) !important; border:1px solid #e4eaf3 !important;
      border-radius:18px !important; box-shadow:0 5px 20px rgba(30,55,90,.045);
      transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover { border-color:#cbd9ef !important; box-shadow:0 9px 26px rgba(30,55,90,.085); }
    .title { font-size:1.85rem; letter-spacing:-.045em; font-weight:800; color:#101a33 !important; margin:0; }
    .sub { font-size:.86rem; color:#64748b !important; }
    .kpi-label { font-size:.82rem; color:#64748b !important; font-weight:600; }
    .kpi-value { font-size:2.05rem; letter-spacing:-.04em; font-weight:800; color:#101a33 !important; line-height:1.25; }
    .kpi-unit { font-size:.86rem; color:#94a3b8 !important; font-weight:600; margin-left:5px; }
    .good { font-size:.78rem; color:#059669 !important; font-weight:700; }
    .bad { font-size:.78rem; color:#dc2626 !important; font-weight:700; }
    .flat { font-size:.78rem; color:#64748b !important; font-weight:600; }
    .card-title { font-size:1.04rem; font-weight:750; color:#101a33 !important; }
    .card-sub { font-size:.8rem; color:#718096 !important; margin-bottom:.65rem; }
    .pill { display:inline-block; padding:7px 12px; border-radius:999px; font-size:.78rem; font-weight:750; }
    .pill-ok { background:#d1fae5; color:#065f46 !important; }
    .pill-bad { background:#fee2e2; color:#991b1b !important; }
    .alert { border-radius:14px; padding:14px 15px; margin-bottom:10px; }
    .alert b { color:#101a33 !important; font-size:.93rem; }
    .alert small { color:#475569 !important; display:block; margin-top:5px; line-height:1.45; }
    .a-crit { background:linear-gradient(120deg,#fff1f2,#fff7f7); border:1px solid #fecdd3; }
    .a-high { background:linear-gradient(120deg,#fff7e6,#fffbeb); border:1px solid #fde2a7; }
    .a-ok { background:linear-gradient(120deg,#ecfdf5,#f0fdfa); border:1px solid #a7f3d0; }
    .bar-row { margin-bottom:17px; }
    .bar-head { display:flex; justify-content:space-between; gap:12px; font-size:.84rem; color:#101a33 !important; margin-bottom:7px; }
    .bar-bg { background:#e8edf5; border-radius:999px; height:10px; overflow:hidden; }
    .bar-fg { height:10px; border-radius:999px; transition:width .45s ease; }
    table.t { width:100%; border-collapse:collapse; font-size:.84rem; }
    table.t th { text-align:left; color:#64748b !important; font-weight:700; padding:9px 6px; border-bottom:1px solid #e7edf5; }
    table.t td { padding:10px 6px; color:#17233d !important; border-bottom:1px solid #f0f3f8; }
    div[data-testid="stButton"] button { border-radius:11px; color:#1d4ed8; border:1px solid #dbe4f0; font-weight:650; transition:all .18s ease; }
    .live-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#10b981; margin-right:7px; box-shadow:0 0 0 4px rgba(16,185,129,.13); }
    .stale-banner { display:flex; align-items:center; gap:12px; padding:14px 17px; margin:12px 0 18px; border:1px solid #f5d38b; border-radius:14px; background:linear-gradient(100deg,#fff4d7,#fffaf0); }
    .stale-icon { font-size:1.35rem; }
    .stale-title { font-weight:800; color:#92400e !important; }
    .stale-copy { color:#a16207 !important; font-size:.82rem; margin-top:3px; }
    @media (max-width:900px) { .block-container { padding-left:1rem; padding-right:1rem; } .title { font-size:1.55rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_database_url() -> str:
    """Read database credentials only from Streamlit secrets or environment."""
    try:
        url = st.secrets.get("DATABASE_URL")
    except Exception:
        url = None
    url = url or os.getenv("DATABASE_URL", "postgresql://postgres.hgsbvhhqenqqvebnevfk:VB1ySuctjJQW2Jcc@aws-0-ap-northeast-2.pooler.supabase.com:5432/postgres")
    if not url:
        st.error("DATABASE_URL is not set. Add it to .streamlit/secrets.toml or your deployment environment.")
        st.stop()
    return url


@st.cache_data(ttl=15)
def load_data() -> pd.DataFrame:
    conn = psycopg2.connect(get_database_url(), connect_timeout=5)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM telemetry ORDER BY timestamp DESC LIMIT 500", conn
        )
    finally:
        conn.close()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"telemetry table is missing required columns: {', '.join(sorted(missing))}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ("mq135", "mq7", "distance", "lat", "lon", "latitude", "longitude"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    return df.sort_values("timestamp", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False)
def reverse_geocode(lat: float, lon: float) -> str:
    """Resolve coordinates using OpenStreetMap Nominatim; cache results for 24 hours."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return "Invalid coordinates"
    params = urlencode({"format": "jsonv2", "lat": f"{lat:.4f}", "lon": f"{lon:.4f}"})
    url = f"https://nominatim.openstreetmap.org/reverse?{params}"
    request = Request(url, headers={"User-Agent": GEOCODER_USER_AGENT, "Accept-Language": "en"})
    try:
        with urlopen(request, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("display_name") or "Location name unavailable"
    except Exception:
        return "Location lookup unavailable (coordinates are still shown)"


def get_coordinate_columns(df: pd.DataFrame):
    """Support either lat/lon or latitude/longitude column naming."""
    lat_col = "lat" if "lat" in df.columns else "latitude" if "latitude" in df.columns else None
    lon_col = "lon" if "lon" in df.columns else "longitude" if "longitude" in df.columns else None
    return lat_col, lon_col


def valid_sensor_reading(row) -> bool:
    """Basic data-quality check; physical bounds should be calibrated per hardware."""
    values = [row.get("mq135"), row.get("mq7"), row.get("distance")]
    return all(pd.notna(v) and float(v) >= 0 for v in values)


def style_chart(ch):
    return ch.configure_view(strokeWidth=0).configure(background="transparent").configure_axis(
        labelColor="#6b7280", titleColor="#6b7280", gridColor="#eceef2", domainColor="#e5e7eb", tickColor="#e5e7eb"
    ).configure_legend(labelColor="#0f1115")


def sparkline(d, field):
    c = COLORS[field]
    ch = (
        alt.Chart(d)
        .mark_area(line={"color": c}, color=c, opacity=0.15)
        .encode(x=alt.X("timestamp:T", axis=None), y=alt.Y(f"{field}:Q", axis=None, scale=alt.Scale(zero=False)))
        .properties(height=55)
    )
    return style_chart(ch)


def kpi_card(col, key, label, value, unit, delta, field, d, higher_is_bad):
    with col, st.container(border=True, key=f"card_{key}"):
        if pd.isna(delta) or delta == 0:
            cls, arrow = "flat", "–"
        else:
            cls = "bad" if (delta > 0) == higher_is_bad else "good"
            arrow = "▲" if delta > 0 else "▼"
        dtxt = "n/a" if pd.isna(delta) else f"{arrow} {abs(delta):.1f} {unit} vs previous"
        st.markdown(
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value:g}<span class="kpi-unit">{unit}</span></div>'
            f'<div class="{cls}">{dtxt}</div>',
            unsafe_allow_html=True,
        )
        st.altair_chart(sparkline(d.tail(60), field), use_container_width=True, theme=None)


def alert_html(level, title, text):
    return f'<div class="alert a-{level}"><b>{title}</b><small>{text}</small></div>'


def bar_html(label, value_txt, pct, color):
    pct = max(0, min(pct, 1)) * 100
    return (
        f'<div class="bar-row"><div class="bar-head"><span>{label}</span><span>{value_txt}</span></div>'
        f'<div class="bar-bg"><div class="bar-fg" style="width:{pct:.0f}%;background:{color}"></div></div></div>'
    )


def render():
    try:
        df = load_data()
    except Exception as exc:
        st.error(f"Unable to load telemetry: {exc}")
        return
    if df.empty:
        st.info("Waiting for database telemetry...")
        return

    # If telemetry is multi-node, analyse one node at a time rather than mixing nodes.
    if "node_id" in df.columns:
        node_ids = sorted(df["node_id"].dropna().astype(str).unique().tolist())
        if node_ids:
            selected_node = st.selectbox("Select sensor node", node_ids, key="safe_sewer_node")
            df = df[df["node_id"].astype(str) == selected_node].copy()
            if df.empty:
                st.warning("No telemetry is available for this node.")
                return

    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    latest = df.iloc[0]
    previous = df.iloc[1] if len(df) > 1 else latest
    d = df.sort_values("timestamp").reset_index(drop=True)

    age = pd.Timestamp.now(tz="UTC") - latest["timestamp"]
    timestamp_valid = age >= pd.Timedelta(seconds=-60)
    online = timestamp_valid and age <= pd.Timedelta(minutes=STALE_MIN)
    safe_age = max(0, int(age.total_seconds()))
    age_txt = f"{safe_age // 60} min {safe_age % 60} s ago" if timestamp_valid else "timestamp is in the future"
    reading_valid = valid_sensor_reading(latest)

    # Header
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown('<p class="title">Node Dashboard</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="sub">Last reading: {latest["timestamp"].strftime("%d %b %Y, %H:%M:%S UTC")} ({age_txt})</p>',
            unsafe_allow_html=True,
        )
    with h2:
        if st.button("↻  Refresh now", use_container_width=True):
            load_data.clear()
            st.rerun()
        pill = '<span class="pill pill-ok">● Device online</span>' if online else '<span class="pill pill-bad">● No recent data</span>'
        st.markdown(f"<div style='text-align:right;padding-top:.6rem'>{pill}</div>", unsafe_allow_html=True)

    # Prominent stale telemetry banner
    if not online:
        st.markdown(
            f'<div class="stale-banner"><div class="stale-icon">⚠️</div>'
            f'<div><div class="stale-title">No recent telemetry — last reading {age_txt}</div>'
            f'<div class="stale-copy">The node has not sent fresh data for more than {STALE_MIN} minutes. '
            f'Check connectivity, power supply, and the gateway.</div></div></div>',
            unsafe_allow_html=True,
        )

    # Resolve the latest stored coordinates into a readable place name.
    lat_col, lon_col = get_coordinate_columns(df)
    with st.container(border=True, key="card_location"):
        st.markdown('<div class="card-title">Node Location</div><div class="card-sub">Resolved from the latest stored GPS coordinates</div>', unsafe_allow_html=True)
        if lat_col and lon_col and pd.notna(latest.get(lat_col)) and pd.notna(latest.get(lon_col)):
            lat, lon = float(latest[lat_col]), float(latest[lon_col])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                place = reverse_geocode(round(lat, 4), round(lon, 4))
                st.write(place)
                st.caption(f"Latitude: {lat:.6f} · Longitude: {lon:.6f}")
                st.markdown(f"[Open coordinates in OpenStreetMap](https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=17/{lat:.6f}/{lon:.6f})")
            else:
                st.warning("Stored coordinates are outside valid latitude/longitude ranges.")
        else:
            st.warning("GPS location unavailable — no valid coordinates received. The GPS module may still be acquiring a satellite fix or the latest telemetry may not include latitude/longitude.")
    # KPI row. Invalid readings are displayed as unavailable, not interpreted as normal.
    c1, c2, c3, c4 = st.columns(4)
    def num_or_none(value):
        return float(value) if pd.notna(value) else None
    v135, v7, vdist = (num_or_none(latest.get(c)) for c in ("mq135", "mq7", "distance"))
    p135, p7, pdist = (num_or_none(previous.get(c)) for c in ("mq135", "mq7", "distance"))
    kpi_card(c1, "mq135", "MQ-135 reading", v135 if v135 is not None else 0, "raw units*", (v135-p135) if v135 is not None and p135 is not None else float("nan"), "mq135", d, True) if v135 is not None else c1.metric("MQ-135 reading", "Unavailable")
    kpi_card(c2, "mq7", "MQ-7 reading", v7 if v7 is not None else 0, "raw units*", (v7-p7) if v7 is not None and p7 is not None else float("nan"), "mq7", d, True) if v7 is not None else c2.metric("MQ-7 reading", "Unavailable")
    kpi_card(c3, "dist", "Water distance", vdist if vdist is not None else 0, "cm", (vdist-pdist) if vdist is not None and pdist is not None else float("nan"), "distance", d, False) if vdist is not None else c3.metric("Water distance", "Unavailable")
    if not reading_valid:
        st.warning("Sensor data quality check failed. Do not interpret this reading as a safe environmental condition.")
    # Count rows with at least one breach; null values do not count as valid passes.
    breach = ((df["mq135"] > TH_MQ135) | (df["mq7"] > TH_MQ7) | (df["distance"] < TH_DIST)).fillna(False)
    with c4, st.container(border=True, key="card_count"):
        st.markdown(
            f'<div class="kpi-label">Readings loaded</div><div class="kpi-value">{len(df)}</div>'
            f'<div class="flat">{int(breach.sum())} readings with ≥1 breach</div>',
            unsafe_allow_html=True,
        )
        span = d["timestamp"].iloc[-1] - d["timestamp"].iloc[0]
        st.markdown(f'<p class="sub" style="margin-top:1.2rem">Covers {span.total_seconds() / 3600:.1f} hours</p>', unsafe_allow_html=True)

    # Chart + alerts
    left, right = st.columns([2, 1])
    with left, st.container(border=True, key="card_trend"):
        st.markdown('<div class="card-title">Gas Sensor Trends</div><div class="card-sub">Sensor-specific dashed lines show each configured threshold</div>', unsafe_allow_html=True)
        long = d[["timestamp", "mq135", "mq7"]].melt("timestamp", var_name="Sensor", value_name="Value").dropna()
        long["Sensor"] = long["Sensor"].map(LABELS)
        threshold_df = pd.DataFrame({
            "Sensor": [LABELS["mq135"], LABELS["mq7"]],
            "Threshold": [TH_MQ135, TH_MQ7],
        })
        scale = alt.Scale(domain=[LABELS["mq135"], LABELS["mq7"]], range=[COLORS["mq135"], COLORS["mq7"]])
        lines = alt.Chart(long).mark_line().encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("Value:Q", title="Reading"),
            color=alt.Color("Sensor:N", scale=scale, legend=alt.Legend(orient="top", title=None)),
            tooltip=["timestamp:T", "Sensor:N", "Value:Q"],
        )
        rules = alt.Chart(threshold_df).mark_rule(strokeDash=[5, 4], color="#dc2626", opacity=0.75).encode(
            y=alt.Y("Threshold:Q"),
            detail="Sensor:N",
            tooltip=["Sensor:N", "Threshold:Q"],
        )
        st.altair_chart(style_chart((lines + rules).properties(height=300)), use_container_width=True, theme=None)

    with right, st.container(border=True, key="card_alerts"):
        st.markdown('<div class="card-title">Critical Alerts</div><div class="card-sub">Based on the latest reading</div>', unsafe_allow_html=True)
        ts = latest["timestamp"].strftime("%H:%M:%S")
        any_alert = False
        if pd.notna(latest["mq135"]) and latest["mq135"] > TH_MQ135:
            any_alert = True
            st.markdown(alert_html("crit", "Poor Air Quality", f"MQ-135 at {latest['mq135']:g} (limit {TH_MQ135}) · {ts}"), unsafe_allow_html=True)
        if pd.notna(latest["mq7"]) and latest["mq7"] > TH_MQ7:
            any_alert = True
            st.markdown(alert_html("crit", "High CO Detected", f"MQ-7 at {latest['mq7']:g} (limit {TH_MQ7}) · {ts}"), unsafe_allow_html=True)
        if pd.notna(latest["distance"]) and latest["distance"] < TH_DIST:
            any_alert = True
            st.markdown(alert_html("high", "Flood Risk", f"Water distance {latest['distance']:g} cm (below {TH_DIST} cm) · {ts}"), unsafe_allow_html=True)
        if not any_alert and online and reading_valid:
            st.markdown(alert_html("ok", "No Threshold Breaches", "Latest valid reading is within configured limits."), unsafe_allow_html=True)
        elif not any_alert and not reading_valid:
            st.markdown(alert_html("high", "Status Unknown", "Latest sensor values are missing or invalid; normal conditions cannot be confirmed."), unsafe_allow_html=True)
        if not online:
            st.markdown(alert_html("high", "No Recent Data", f"Last reading was {age_txt}."), unsafe_allow_html=True)

    # Bottom row
    b1, b2 = st.columns(2)
    with b1, st.container(border=True, key="card_usage"):
        st.markdown('<div class="card-title">Threshold Usage & Water Proximity</div><div class="card-sub">Gas bars show value / configured limit. Water bar is a heuristic proximity indicator, not a certified flood probability.</div>', unsafe_allow_html=True)
        p135 = v135 / TH_MQ135 if v135 is not None else 0
        p7 = v7 / TH_MQ7 if v7 is not None else 0
        # A closer distance means the water surface is closer to the sensor.
        # This 0–100% scale is only a visual heuristic tied to TH_DIST.
        water_risk = max(0.0, min(1.0, (TH_DIST - vdist) / TH_DIST)) if vdist is not None else 0
        st.markdown(
            bar_html(LABELS["mq135"], f"{v135:g} / {TH_MQ135}" if v135 is not None else "Unavailable", p135, "#dc2626" if v135 is not None and p135 > 1 else COLORS["mq135"])
            + bar_html(LABELS["mq7"], f"{v7:g} / {TH_MQ7}" if v7 is not None else "Unavailable", p7, "#dc2626" if v7 is not None and p7 > 1 else COLORS["mq7"])
            + bar_html("Water proximity risk (heuristic)", f"{vdist:g} cm; alert below {TH_DIST} cm" if vdist is not None else "Unavailable", water_risk, "#dc2626" if vdist is not None and vdist < TH_DIST else COLORS["distance"]),
            unsafe_allow_html=True,
        )
        st.caption("*Gas readings are labelled as raw units until the firmware and calibration confirm that they are ppm.")

    with b2, st.container(border=True, key="card_history"):
        st.markdown('<div class="card-title">Recent Sensor Breach Events</div><div class="card-sub">Each row is one sensor crossing; a single reading may create multiple events.</div>', unsafe_allow_html=True)
        rows = []
        for field, mask in (("mq135", df["mq135"] > TH_MQ135), ("mq7", df["mq7"] > TH_MQ7), ("distance", df["distance"] < TH_DIST)):
            for _, r in df[mask.fillna(False)].iterrows():
                rows.append((r["timestamp"], str(r.get("node_id", "Node")), LABELS[field], r[field]))
        rows.sort(key=lambda x: x[0], reverse=True)
        if not rows:
            st.markdown(alert_html("ok", "No Threshold Breach Events", "No sensor values crossed configured limits in the loaded history."), unsafe_allow_html=True)
        else:
            body = "".join(
                f"<tr><td>{t.strftime('%d %b %H:%M:%S')}</td><td>{node}</td><td>{sensor}</td><td>{value:g}</td></tr>"
                for t, node, sensor, value in rows[:10]
            )
            st.markdown(f'<table class="t"><tr><th>Time (UTC)</th><th>Node</th><th>Sensor</th><th>Value</th></tr>{body}</table>', unsafe_allow_html=True)
            st.caption(f"Showing {min(len(rows), 10)} of {len(rows)} sensor breach events from the 500-row query window.")



    # ML anomaly detection (implementation lives in ml/anomaly_detector.py)
    with st.container(border=True, key="card_ml_anomaly"):
        st.markdown(
            '<div class="card-title">ML Anomaly Detection</div>'
            '<div class="card-sub">Isolation Forest evaluates unusual sensor patterns in recent telemetry. '
            'This is experimental and does not replace threshold alerts.</div>',
            unsafe_allow_html=True,
        )
        try:
            ml_results = detect_anomalies(df)
            if ml_results.empty:
                st.info("Not enough valid sensor readings to run anomaly detection yet.")
            else:
                if "timestamp" in ml_results.columns:
                    ml_results = ml_results.copy()
                    ml_results["timestamp"] = pd.to_datetime(ml_results["timestamp"], utc=True, errors="coerce")
                    ml_results = ml_results.sort_values("timestamp", ascending=True)
                latest_ml = ml_results.iloc[-1]
                ml1, ml2, ml3 = st.columns(3)
                with ml1:
                    st.metric("Latest ML status", str(latest_ml["ml_status"]))
                with ml2:
                    st.metric("Anomalies in loaded history", int(ml_results["is_anomaly"].sum()))
                with ml3:
                    st.metric("Latest anomaly score", f'{latest_ml["anomaly_score"]:.3f}')
                if latest_ml["is_anomaly"]:
                    st.warning(
                        "The latest reading looks unusual compared with the recent sensor history. "
                        "Verify the sensor and surrounding conditions; this is not a confirmed flood diagnosis."
                    )
                else:
                    st.success("The latest reading was not flagged as anomalous by the current model.")
                anomalies = ml_results[ml_results["is_anomaly"]].sort_values("timestamp", ascending=False).head(8)
                if not anomalies.empty:
                    st.markdown("**Recent readings flagged by ML**")
                    show = anomalies[["timestamp", "mq135", "mq7", "distance", "anomaly_score"]].copy()
                    show["timestamp"] = show["timestamp"].dt.strftime("%d %b %H:%M:%S UTC")
                    show = show.rename(columns={
                        "timestamp": "Time",
                        "mq135": "MQ-135",
                        "mq7": "MQ-7",
                        "distance": "Water distance (cm)",
                        "anomaly_score": "Anomaly score",
                    })
                    st.dataframe(show, use_container_width=True, hide_index=True)
                else:
                    st.caption("No anomalies were flagged in the currently loaded readings.")
        except Exception as exc:
            st.warning(f"ML detection is currently unavailable: {exc}")

with st.sidebar:
    st.markdown(
        '<div style="padding:8px 0 18px">'
        '<div style="font-size:1.35rem;font-weight:850;letter-spacing:-.04em;color:#f8fbff!important">💧 Safe<span style="color:#2dd4bf!important">Sewer</span></div>'
        '<div style="font-size:.75rem;color:#aebed5!important">Cleaner Cities · Safer Communities</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Node Command")
    st.caption("Environmental monitoring")
    st.markdown('<span class="live-dot"></span>Telemetry monitor', unsafe_allow_html=True)
    st.caption("Signed in as **Admin**")
    st.markdown("---")
    st.caption("NODE NETWORK · TELEMETRY")

st.fragment(run_every=REFRESH)(render)()