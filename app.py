# app.py — Step 1 (clean): single request, 3–5 candidate providers, 3D two-layer view
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass

st.set_page_config(page_title="DT Request Demo — Step 1", layout="wide")

# -------------------- Types & State -------------------- #
@dataclass
class Scene:
    n: int
    mal_pct: int                # % malicious nodes
    seed: int
    requester: int
    candidates: list[int]
    selected: int | None
    t_true: float | None
    service_value: float | None
    requested: bool
    evaluated: bool
    malicious: set[int]

# -------------------- Graph & Rendering -------------------- #
def build_layers_3d(n=20, seed=42, k=4, rewire_p=0.25, z_sep=1):
    """
    DT graph (Watts–Strogatz) and two 3D layers:
    - Real devices: z = 0
    - Digital Twins: z = z_sep
    """
    Gdt = nx.watts_strogatz_graph(n=n, k=min(k, max(2, n//3)), p=rewire_p, seed=int(seed))
    pos2d = nx.spring_layout(Gdt, seed=int(seed), dim=2)

    xs = np.array([pos2d[i][0] for i in range(n)], dtype=float)
    ys = np.array([pos2d[i][1] for i in range(n)], dtype=float)

    # Normalize to [-1, 1] (NumPy 2.0-safe)
    ptp_x = float(np.ptp(xs));  ptp_y = float(np.ptp(ys))
    if ptp_x > 0: xs = 2.0 * (xs - float(np.min(xs))) / ptp_x - 1.0
    if ptp_y > 0: ys = 2.0 * (ys - float(np.min(ys))) / ptp_y - 1.0

    pos_dt3d = {i: (float(xs[i]), float(ys[i]), float(z_sep)) for i in range(n)}
    pos_rl3d = {i: (float(xs[i]), float(ys[i]), 0.0) for i in range(n)}
    return Gdt, pos_rl3d, pos_dt3d

def fig_two_layers_3d(Gdt, pos_rl3d, pos_dt3d, requester, candidates, selected):
    """
    3D scene:
    - DT↔DT: blue lines (upper plane)
    - Real↔DT: light-gray dashed vertical links (fine & dense)
    - Real nodes larger; DT nodes color-coded for role
    """
    fig = go.Figure()

    # DT edges (continuous)
    x_e, y_e, z_e = [], [], []
    for u, v in Gdt.edges():
        x0, y0, z0 = pos_dt3d[u]; x1, y1, z1 = pos_dt3d[v]
        x_e += [x0, x1, None]; y_e += [y0, y1, None]; z_e += [z0, z1, None]
    fig.add_trace(go.Scatter3d(
        x=x_e, y=y_e, z=z_e, mode="lines",
        line=dict(width=4, color="#9ec9ff"), opacity=0.6,
        hoverinfo="skip", name="DT links"
    ))

    # Real↔DT dashed verticals (dense & thin)
    xv, yv, zv = [], [], []
    n_dashes = 60
    duty     = 0.45
    for i in pos_rl3d:
        x0, y0, z0 = pos_rl3d[i]; x1, y1, z1 = pos_dt3d[i]
        zmin, zmax = (z0, z1) if z1 >= z0 else (z1, z0)
        dz = (zmax - zmin) / n_dashes
        for k in range(n_dashes):
            zs = zmin + k*dz
            ze = zmin + (k + duty)*dz
            xv += [x0, x0, None];  yv += [y0, y0, None];  zv += [zs, ze, None]
    fig.add_trace(go.Scatter3d(
        x=xv, y=yv, z=zv, mode="lines",
        line=dict(width=2, color="#c8c8c8"), opacity=0.9,
        hoverinfo="skip", name="Real→DT (dashed)"
    ))

    # Real nodes (bigger)
    xr = [pos_rl3d[i][0] for i in pos_rl3d]
    yr = [pos_rl3d[i][1] for i in pos_rl3d]
    zr = [pos_rl3d[i][2] for i in pos_rl3d]
    fig.add_trace(go.Scatter3d(
        x=xr, y=yr, z=zr, mode="markers",
        marker=dict(size=7, color="#888888"),
        hovertemplate="Real device #%{customdata}<extra></extra>",
        customdata=list(pos_rl3d.keys()),
        name="Real"
    ))

    # DT nodes (role-based style)
    xd = []; yd = []; zd = []; colors = []; sizes = []; labels = []
    for i, (x, y, z) in pos_dt3d.items():
        if i == requester:
            colors.append("#FFD166"); sizes.append(16); label = f"DT #{i} (Requester)"
        elif selected is not None and i == selected:
            colors.append("#EF476F"); sizes.append(18); label = f"DT #{i} (Selected)"
        elif i in candidates:
            colors.append("#06D6A0"); sizes.append(14); label = f"DT #{i} (Candidate)"
        else:
            colors.append("#118AB2"); sizes.append(12); label = f"DT #{i}"
        xd.append(x); yd.append(y); zd.append(z); labels.append(label)
    fig.add_trace(go.Scatter3d(
        x=xd, y=yd, z=zd, mode="markers",
        marker=dict(size=sizes, color=colors, line=dict(width=1.2, color="#222")),
        hovertemplate="%{text}<extra></extra>", text=labels,
        name="Digital Twins"
    ))

    # Calcolo il range Z dai punti (così non devo passare z_sep)
    z_vals = [p[2] for p in pos_rl3d.values()] + [p[2] for p in pos_dt3d.values()]

    fig.update_layout(
        height=650, margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False, range=[min(z_vals)-0.02, max(z_vals)+0.02]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.3),  # z più piccolo = layer più vicini visivamente
            camera=dict(
                eye=dict(x=0.6, y=1, z=0.6),
                center=dict(x=0.0, y=0.0, z=-0.2))
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02, x=0.02,
            font=dict(size=16)   # <-- dimensione testo legenda
        )
    )
    return fig

# -------------------- Simple environment & service -------------------- #
def simulate_temperature_field(n, seed=42):
    rng = np.random.default_rng(seed + 200)
    base = 22 + rng.normal(0, 2, 1)[0]      # ambient baseline
    spatial = np.linspace(-0.8, 0.8, n)     # simple gradient
    return base, spatial

def provider_reading(t_true, provider_id, malicious: set[int], seed=42):
    rng = np.random.default_rng(seed + 300 + provider_id)
    if provider_id in malicious:
        # Malicious: "a couple of degrees" off (+/- ~2 ± 0.5 °C)
        sign = 1 if rng.random() < 0.5 else -1
        offset = 2.0 + rng.normal(0, 0.5)
        return float(t_true + sign * offset)
    else:
        # Benevolent: very close to truth
        return float(t_true + rng.normal(0, 0.25))

# -------------------- Init state -------------------- #
if "scene" not in st.session_state:
    st.session_state.scene = Scene(
        n=20, mal_pct=20, seed=42,
        requester=0, candidates=[], selected=None,
        t_true=None, service_value=None,
        requested=False, evaluated=False,
        malicious=set()
    )
S: Scene = st.session_state.scene

# -------------------- Sidebar (ONLY what you asked) -------------------- #
with st.sidebar:
    st.header("Scenario")
    S.n = st.slider("Number of nodes", 8, 30, S.n, 1)
    S.mal_pct = st.slider("Malicious nodes (%)", 0, 100, S.mal_pct, 5)
    st.caption("Step 1: No trust, only service discovery and evaluation.")

# -------------------- Build world & malicious set -------------------- #
Gdt, pos_rl3d, pos_dt3d = build_layers_3d(n=S.n, seed=S.seed)
# Recompute malicious set if needed
rng_global = np.random.default_rng(S.seed + 7)
k_mal = int(round(S.n * (S.mal_pct / 100.0)))
if len(S.malicious) != k_mal or any(i >= S.n for i in S.malicious):
    S.malicious = set(rng_global.choice(list(range(S.n)), size=k_mal, replace=False)) if k_mal > 0 else set()

# Reset selections if graph size changed
if S.requester >= S.n:
    S.requester = 0
    S.candidates = []
    S.selected = None
    S.requested = False
    S.evaluated = False
    S.service_value = None
    S.t_true = None

# -------------------- Layout -------------------- #
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Digital Twins network")
    fig = fig_two_layers_3d(Gdt, pos_rl3d, pos_dt3d, S.requester, S.candidates, S.selected)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Service Request")
    # Requester picker (moved to the right)
    S.requester = st.selectbox(
        "Requester",
        options=list(range(S.n)),
        index=S.requester,
        format_func=lambda i: f"DT {i}"
    )

    # Candidate selection
    neighbors = list(Gdt.neighbors(S.requester))
    pool = neighbors if len(neighbors) >= 3 else list(set(range(S.n)) - {S.requester})
    if st.button("Discovery Providers"):
        m = int(rng_global.integers(3, 6))  # 3..5 providers
        m = min(m, max(1, len(pool)))
        S.candidates = list(rng_global.choice(pool, size=m, replace=False)) if pool else []
        S.selected = None
        S.requested = False
        S.evaluated = False
        S.service_value = None
        S.t_true = None

    if S.candidates:
        S.selected = st.radio("Choose one provider", options=S.candidates, format_func=lambda i: f"DT #{i}")

    st.markdown("---")
    st.subheader("Interaction")

    # Gate 1: show "Request service" only after selecting a provider
    if S.selected is not None:
        base, spatial = simulate_temperature_field(S.n, seed=S.seed)
        if st.button("Request service"):
            S.t_true = float(base + spatial[S.requester] + np.random.normal(0, 0.2))
            S.service_value = provider_reading(S.t_true, S.selected, S.malicious, seed=S.seed)
            S.requested = True
            S.evaluated = False

    # Show service value if requested
    if S.requested and S.service_value is not None and S.selected is not None:
        st.metric(f"Service value from DT {S.selected}", f"{S.service_value:.1f} °C")

        # Gate 2: Evaluate after requesting
        if st.button("Evaluate Service"):
            S.evaluated = True

    # Final evaluation (Yes/No based on maliciousness)
    if S.evaluated and S.selected is not None:
        good = (S.selected not in S.malicious)
        if good:
            st.success("Service evaluation: Correct Service!")
        else:
            st.error("Service evaluation: Wrong Service!")


st.caption("Step 1: No trust, only service discovery and evaluation.")
