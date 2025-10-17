# app.py — DT Request Demo (no sidebar) — 30 nodes, per-service providers & bad sets, radio wrapped, stable plot
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass, field

st.set_page_config(page_title="DT Request Demo — Step 1", layout="wide")

# Hide Streamlit top header + kill any horizontal scrollbar + disable clickable headings
st.markdown("""
<style>
/* nascondi header Streamlit */
header[data-testid="stHeader"] { display: none !important; }
.block-container { padding-top: 0.5rem; }

/* niente scrollbar orizzontale globale */
html, body, [data-testid="stAppViewContainer"] { overflow-x: hidden !important; }

/* -- Rendi NON cliccabili i titoli (rimuovi i permalinks) -- */
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a,
[data-testid="stMarkdownContainer"] h1 a,
[data-testid="stMarkdownContainer"] h2 a,
[data-testid="stMarkdownContainer"] h3 a,
[data-testid="stMarkdownContainer"] h4 a,
[data-testid="stMarkdownContainer"] h5 a,
[data-testid="stMarkdownContainer"] h6 a {
  pointer-events: none !important;   /* disabilita click */
  color: inherit !important;         /* niente colore "link" */
  text-decoration: none !important;  /* niente sottolineatura */
}
/* opzionale: nascondi l'iconcina del link se presente */
h1 a svg, h2 a svg, h3 a svg, h4 a svg, h5 a svg, h6 a svg { display: none !important; }
</style>
""", unsafe_allow_html=True)

SERVICES = ["Temperature", "Humidity", "Power"]

# -------------------- State -------------------- #
@dataclass
class Scene:
    n: int = 30                 # fisso a 30 nodi
    seed: int = 42
    created: bool = False
    Gdt: nx.Graph | None = None
    pos_rl3d: dict = field(default_factory=dict)
    pos_dt3d: dict = field(default_factory=dict)
    avg_deg_target: int = 4
    # per-service: providers (30–50%) e bad responders (30%) — DEVONO essere diversi per servizio
    providers_by_service: dict = field(default_factory=lambda: {s: set() for s in SERVICES})
    bad_by_service: dict = field(default_factory=lambda: {s: set() for s in SERVICES})
    requester: int = 0
    service: str = "Temperature"
    providers: list[int] = field(default_factory=list)     # candidati per il servizio selezionato
    selected: int | None = None
    evaluated: bool = False
    readings: dict = field(default_factory=dict)           # (service, provider) -> value
    last_plot_key: tuple | None = None                     # evita ridisegno inutile

if "scene" not in st.session_state:
    st.session_state.scene = Scene()
S: Scene = st.session_state.scene

# --- unique keys per i chart (evita conflitti Streamlit) --- #
if "_graph_key_counter" not in st.session_state:
    st.session_state._graph_key_counter = 0

def _next_graph_key(tag="g"):
    st.session_state._graph_key_counter += 1
    return f"{tag}-{S.seed}-{st.session_state._graph_key_counter}"

# -------------------- Graph helpers -------------------- #
def build_layers_3d(n=30, seed=42, target_avg_deg=4, z_sep=0.6):
    """Grafo ER connesso (grado medio ≈ 3–5) e posizioni 3D per Real(z=0) & DT(z=z_sep)."""
    rng = np.random.default_rng(seed)
    p = min(1.0, float(target_avg_deg) / max(1, (n - 1)))

    # Tenta più ER; se ancora disconnesso, collega componenti con pochi edge extra
    for _ in range(6):
        Gdt = nx.erdos_renyi_graph(n=n, p=p, seed=int(rng.integers(0, 1_000_000)))
        if nx.is_connected(Gdt):
            break
    if not nx.is_connected(Gdt):
        comps = [list(c) for c in nx.connected_components(Gdt)]
        for a, b in zip(comps, comps[1:]):
            u = int(rng.choice(a)); v = int(rng.choice(b))
            Gdt.add_edge(u, v)

    pos2d = nx.spring_layout(Gdt, seed=int(seed), dim=2)
    xs = np.array([pos2d[i][0] for i in range(n)], dtype=float)
    ys = np.array([pos2d[i][1] for i in range(n)], dtype=float)
    ptp_x = float(np.ptp(xs));  ptp_y = float(np.ptp(ys))
    if ptp_x > 0: xs = 2.0 * (xs - float(np.min(xs))) / ptp_x - 1.0
    if ptp_y > 0: ys = 2.0 * (ys - float(np.min(ys))) / ptp_y - 1.0
    pos_dt3d = {i: (float(xs[i]), float(ys[i]), float(z_sep)) for i in range(n)}
    pos_rl3d = {i: (float(xs[i]), float(ys[i]), 0.0) for i in range(n)}
    return Gdt, pos_rl3d, pos_dt3d

def fig_two_layers_3d(Gdt, pos_rl3d, pos_dt3d, requester, candidates, selected):
    """Scena statica: DT↔DT (azzurro), Real↔DT verticali tratteggiate, nodi con colori di ruolo."""
    fig = go.Figure()

    # DT edges
    x_e, y_e, z_e = [], [], []
    for u, v in Gdt.edges():
        x0, y0, z0 = pos_dt3d[u]; x1, y1, z1 = pos_dt3d[v]
        x_e += [x0, x1, None]; y_e += [y0, y1, None]; z_e += [z0, z1, None]
    fig.add_trace(go.Scatter3d(
        x=x_e, y=y_e, z=z_e, mode="lines",
        line=dict(width=4, color="#9ec9ff"), opacity=0.6, hoverinfo="skip", name="DT links"
    ))

    # Vertical dashed
    xv, yv, zv = [], [], []
    n_dashes = 60; duty = 0.45
    for i in pos_rl3d:
        x0, y0, z0 = pos_rl3d[i]; x1, y1, z1 = pos_dt3d[i]
        zmin, zmax = (z0, z1) if z1 >= z0 else (z1, z0)
        dz = (zmax - zmin) / n_dashes
        for k in range(n_dashes):
            zs = zmin + k*dz; ze = zmin + (k + duty)*dz
            xv += [x0, x0, None];  yv += [y0, y0, None];  zv += [zs, ze, None]
    fig.add_trace(go.Scatter3d(
        x=xv, y=yv, z=zv, mode="lines",
        line=dict(width=2, color="#c8c8c8"), opacity=0.9, hoverinfo="skip", name="Real→DT (dashed)"
    ))

    # Real nodes
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

    # DT nodes (colori per ruolo)
    xd, yd, zd, colors, sizes, labels = [], [], [], [], [], []
    for i, (x, y, z) in pos_dt3d.items():
        if i == requester:
            colors.append("#FFD166"); sizes.append(16); label = f"DT {i} (Requester)"
        elif selected is not None and i == selected:
            colors.append("#EF476F"); sizes.append(18); label = f"DT {i} (Selected)"
        elif i in candidates:
            colors.append("#06D6A0"); sizes.append(14); label = f"DT {i} (Candidate)"
        else:
            colors.append("#118AB2"); sizes.append(12); label = f"DT {i}"
        xd.append(x); yd.append(y); zd.append(z); labels.append(label)
    fig.add_trace(go.Scatter3d(
        x=xd, y=yd, z=zd, mode="markers",
        marker=dict(size=sizes, color=colors, line=dict(width=1.2, color="#222")),
        hovertemplate="%{text}<extra></extra>", text=labels, name="Digital Twins"
    ))

    # Layout
    z_vals = ([p[2] for p in pos_rl3d.values()] + [p[2] for p in pos_dt3d.values()]) if pos_dt3d else [0, 1]
    fig.update_layout(
        uirevision="keep",  # conserva zoom/camera
        showlegend=False,
        height=650, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False, range=[min(z_vals)-0.02, max(z_vals)+0.02]),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.3),
            camera=dict(eye=dict(x=0.9, y=1.0, z=0.6))
        )
    )
    return fig

# -------------------- Valori dei servizi -------------------- #
def provider_reading(
    service: str,
    provider_id: int,
    bad_by_service: dict[str, set[int]],
    rng: np.random.Generator
) -> float:
    """
    Valori per servizio:
      Temperature:  TRUE 25–27 °C    | FALSE ~20±0.5 o ~30±0.5
      Humidity:     TRUE 45–50 %     | FALSE ~20±2   o ~90±2
      Power:        TRUE 20–25 W     | FALSE ~-100±50 o ~3000±50
    """
    is_bad = provider_id in bad_by_service.get(service, set())

    if service == "Temperature":
        if is_bad:
            center = 20.0 if rng.random() < 0.5 else 30.0
            return float(center + rng.normal(0, 0.5))
        else:
            return float(rng.uniform(25.0, 27.0))

    elif service == "Humidity":
        if is_bad:
            center = 20.0 if rng.random() < 0.5 else 90.0
            return float(center + rng.normal(0, 2.0))
        else:
            return float(rng.uniform(45.0, 50.0))

    elif service == "Power":
        if is_bad:
            center = -100.0 if rng.random() < 0.5 else 3000.0
            return float(center + rng.normal(0, 50.0))
        else:
            return float(rng.uniform(20.0, 25.0))

    return float("nan")

# -------------------- Init una tantum -------------------- #
def ensure_network_ready():
    if not S.created:
        # usa un seed base, poi derivate diverse per ogni servizio => set diversi garantiti
        base_seed =  int(np.random.default_rng().integers(0, 10**9))
        S.seed = base_seed

        # grafo
        S.avg_deg_target = 4
        S.Gdt, S.pos_rl3d, S.pos_dt3d = build_layers_3d(
            n=S.n, seed=S.seed, target_avg_deg=S.avg_deg_target, z_sep=0.6
        )

        # per-service: providers (30–50%) e bad (30%), con SEED DIVERSO PER SERVIZIO
        nodes = np.arange(S.n)
        S.providers_by_service = {}
        S.bad_by_service = {}

        m_min = int(np.ceil(0.30 * S.n))
        m_max = int(np.floor(0.50 * S.n))
        bad_k = int(round(0.30 * S.n))

        for idx, srv in enumerate(SERVICES):
            # provider: 30–50% con seed dedicato per servizio
            rng_prov = np.random.default_rng(S.seed + 1000 + idx)
            m = int(rng_prov.integers(m_min, m_max + 1))
            prov = set(map(int, rng_prov.choice(nodes, size=m, replace=False)))

            # bad: 30% con altro seed specifico per servizio
            rng_bad = np.random.default_rng(S.seed + 2000 + idx)
            bad = set(map(int, rng_bad.choice(nodes, size=bad_k, replace=False))) if bad_k > 0 else set()

            S.providers_by_service[srv] = prov
            S.bad_by_service[srv] = bad

        S.created = True

ensure_network_ready()

# -------------------- Layout -------------------- #
left, right = st.columns([1.2, 1])

with left:
    st.title("Digital Twins Network")

    # Legenda statica in UNA riga
    st.markdown(
        """
        <div style="display:flex; gap:28px; align-items:center; flex-wrap:nowrap; overflow-x:auto; padding:4px 2px 8px 2px;">
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="display:inline-block;width:12px;height:12px;border-radius:50%; background:#118AB2;border:1.2px solid #222;"></span>
            <span style="font-size:0.95rem;">DT (generic)</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="display:inline-block;width:12px;height:12px;border-radius:50%; background:#FFD166;border:1.2px solid #222;"></span>
            <span style="font-size:0.95rem;">Requester</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="display:inline-block;width:12px;height:12px;border-radius:50%; background:#06D6A0;border:1.2px solid #222;"></span>
            <span style="font-size:0.95rem;">Candidate</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="display:inline-block;width:12px;height:12px;border-radius:50%; background:#EF476F;border:1.2px solid #222;"></span>
            <span style="font-size:0.95rem;">Selected</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # CSS: radio providers su più colonne (wrapping)
    st.markdown(
        """
        <style>
        div[data-testid="stRadio"] > div {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem 0.75rem;
        }
        div[data-testid="stRadio"] label {
            min-width: 90px;  /* aumenta/diminuisci per +/− colonne */
            padding: 2px 2px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    graph_placeholder = st.empty()

def current_plot_key():
    return (
        "draw",
        S.Gdt.number_of_edges() if S.Gdt is not None else 0,
        S.requester,
        tuple(sorted(S.providers)),
        S.selected
    )

def draw_graph_if_needed(force=False):
    key_state = current_plot_key()
    if force or (S.last_plot_key != key_state):
        graph_placeholder.empty()
        fig = fig_two_layers_3d(S.Gdt, S.pos_rl3d, S.pos_dt3d, S.requester, S.providers, S.selected)
        graph_placeholder.plotly_chart(fig, use_container_width=True, key=_next_graph_key("final"), config={"displayModeBar": False})
        S.last_plot_key = current_plot_key()

# Primo draw
draw_graph_if_needed(force=True)

# -------------------- Right: controls -------------------- #
with right:
    st.subheader("Service Request Scenario")

    # Requester & Service
    S.requester = st.selectbox(
        "Requester",
        options=list(range(S.n)),
        index=min(S.requester, S.n-1),
        format_func=lambda i: f"DT {i}",
    )

    S.service = st.selectbox(
        "Service",
        SERVICES,
        index=SERVICES.index(S.service) if S.service in SERVICES else 0
    )

    # Aggiorna candidati per il servizio selezionato (fissi 30–50%), escludendo il requester
    base_set = set(S.providers_by_service.get(S.service, set()))
    base_set.discard(S.requester)
    new_providers = sorted(base_set) if base_set else []
    if new_providers != S.providers:
        S.providers = new_providers
        if S.selected not in S.providers:
            S.selected = None
            S.evaluated = False
    draw_graph_if_needed(force=True)

    # Provider via RADIO (niente pulsanti), su più "colonne" grazie al wrapping CSS
    if S.providers:
        st.markdown("**Service Discovery**")
        radio_key = f"providers-radio-{S.service}-{S.requester}"
        sel = st.radio(
            label="Providers",                    # label non vuota
            options=S.providers,
            format_func=lambda i: f"DT {i}",
            horizontal=True,
            key=radio_key,
            label_visibility="collapsed"         # nascosto a schermo, ma niente warning
        )
        if sel != S.selected:
            S.selected = sel
            S.evaluated = False
            draw_graph_if_needed(force=True)

        # Reading per (service, provider) — cache per non cambiare a ogni click
        SERVICE_SEED = {"Temperature": 11, "Humidity": 22, "Power": 33}

        if S.selected is not None:
            key = (S.service, S.selected)
            if key not in S.readings:
                base = SERVICE_SEED.get(S.service, 0)
                rng_read = np.random.default_rng(S.seed + 777 + base + S.selected)
                S.readings[key] = provider_reading(S.service, S.selected, S.bad_by_service, rng_read)
            val = S.readings[key]
            unit = " °C" if S.service == "Temperature" else (" %" if S.service == "Humidity" else " W")
            st.metric(f"{S.service} from DT {S.selected}", f"{val:.1f}{unit}")

            if st.button("Evaluate", key="eval-btn"):
                S.evaluated = True

            if S.evaluated:
                is_bad = S.selected in S.bad_by_service[S.service]
                if is_bad:
                    st.error("Service evaluation: **Incorrect**")
                else:
                    st.success("Service evaluation: **Correct**")
