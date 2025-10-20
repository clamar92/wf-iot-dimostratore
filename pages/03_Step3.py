# app_step3.py — DT Request Demo (no sidebar) — malicious nodes + evaluation by thresholds + reccomendations
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from dataclasses import dataclass, field

PAGE_TAG = "step3"  # prefisso per le chiavi della pagina

st.set_page_config(page_title="DT Request Demo — Step 3", layout="wide")

# Hide Streamlit top header + kill any horizontal scrollbar + disable clickable headings
st.markdown("""
<style>
header[data-testid="stHeader"] { display: none !important; }
.block-container { padding-top: 0.5rem; }
html, body, [data-testid="stAppViewContainer"] { overflow-x: hidden !important; }
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a,
[data-testid="stMarkdownContainer"] h1 a,
[data-testid="stMarkdownContainer"] h2 a,
[data-testid="stMarkdownContainer"] h3 a,
[data-testid="stMarkdownContainer"] h4 a,
[data-testid="stMarkdownContainer"] h5 a,
[data-testid="stMarkdownContainer"] h6 a {
  pointer-events: none !important;
  color: inherit !important;
  text-decoration: none !important;
}
h1 a svg, h2 a svg, h3 a svg, h4 a svg, h5 a svg { display: none !important; }
</style>
""", unsafe_allow_html=True)

SERVICES = ["Temperature", "Humidity", "Power"]

# -------------------- State -------------------- #
@dataclass
class Scene:
    n: int = 30
    seed: int = 42
    created: bool = False
    Gdt: nx.Graph | None = None
    pos_rl3d: dict = field(default_factory=dict)
    pos_dt3d: dict = field(default_factory=dict)
    avg_deg_target: int = 4
    providers_by_service: dict = field(default_factory=lambda: {s: set() for s in SERVICES})
    requester: int = 0
    service: str = "Temperature"
    providers: list[int] = field(default_factory=list)
    selected: int | None = None
    evaluated: bool = False
    last_eval_ok: bool | None = None  # esito ultima valutazione (soglie benevole)
    readings: dict = field(default_factory=dict)  # (service, provider) -> last shown value
    last_plot_key: tuple | None = None
    # Malevoli sul servizio
    malicious_type: dict = field(default_factory=dict)  # node -> "ME" | "DA" | "OOA"
    # OOA counters per (requester, provider)
    ooa_counter: dict = field(default_factory=dict)  # (req, prov) -> int
    # Malevoli nelle raccomandazioni (indipendenti dai malevoli del servizio)
    rec_malicious_type: dict = field(default_factory=dict)  # node -> "BMA" | "BSA" | None

SCENE_KEY = "scene_step3"
if SCENE_KEY not in st.session_state:
    st.session_state[SCENE_KEY] = Scene()
S: Scene = st.session_state[SCENE_KEY]

GRAPH_KEY = f"{PAGE_TAG}-graph"

# -------------------- Helpers -------------------- #
def build_layers_3d(n=30, seed=42, target_avg_deg=4, z_sep=0.6):
    rng = np.random.default_rng(seed)
    p = min(1.0, float(target_avg_deg) / max(1, (n - 1)))
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
    xs = 2.0 * (xs - np.min(xs)) / max(1e-9, np.ptp(xs)) - 1.0
    ys = 2.0 * (ys - np.min(ys)) / max(1e-9, np.ptp(ys)) - 1.0
    pos_dt3d = {i: (float(xs[i]), float(ys[i]), float(z_sep)) for i in range(n)}
    pos_rl3d = {i: (float(xs[i]), float(ys[i]), 0.0) for i in range(n)}
    return Gdt, pos_rl3d, pos_dt3d

def fig_two_layers_3d(Gdt, pos_rl3d, pos_dt3d, requester, candidates, selected):
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
        dz = (zmax - zmin) / n_dashes if (zmax - zmin) > 0 else 0
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
    # DT nodes (ruoli)
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
    fig.update_layout(
        uirevision="keep",
        showlegend=False,
        height=650, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False), aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.3),
            camera=dict(eye=dict(x=0.9, y=1.0, z=0.6))
        )
    )
    return fig

# ------- Value generators ------- #
def _correct_value(service: str, rng: np.random.Generator) -> float:
    if service == "Temperature":
        return float(rng.uniform(25.0, 27.0))
    elif service == "Humidity":
        return float(rng.uniform(45.0, 50.0))
    elif service == "Power":
        return float(rng.uniform(20.0, 25.0))
    return float("nan")

def _wrong_value_ME_ranges(service: str, rng: np.random.Generator) -> float:
    if service == "Temperature":
        return float(rng.uniform(18.0, 22.0))
    elif service == "Humidity":
        return float(rng.uniform(20.0, 30.0))
    elif service == "Power":
        return float(rng.uniform(200.0, 300.0))
    return float("nan")

def provider_value(service: str, requester: int, provider: int, mal_type: str | None, ooa_count: int, rng: np.random.Generator) -> float:
    """
    - ME: sempre sbagliato (range ME).
    - DA: sbagliato solo con metà dei requester (qui: requester pari => sbaglia; dispari => corretto).
    - OOA: pattern 2 correct, 2 wrong, ripetuto per requester (interazione = Evaluate Service).
    - None: corretto.
    """
    if mal_type == "ME":
        return _wrong_value_ME_ranges(service, rng)
    if mal_type == "DA":
        is_wrong = (requester % 2 == 0)
        return _wrong_value_ME_ranges(service, rng) if is_wrong else _correct_value(service, rng)
    if mal_type == "OOA":
        phase = ooa_count % 4  # 0,1 => correct ; 2,3 => wrong
        is_wrong = (phase in (2, 3))
        return _wrong_value_ME_ranges(service, rng) if is_wrong else _correct_value(service, rng)
    return _correct_value(service, rng)

# ------- Evaluation against benign thresholds ------- #
def is_value_benign(service: str, value: float) -> bool:
    if service == "Temperature":
        return 25.0 <= value <= 27.0
    if service == "Humidity":
        return 45.0 <= value <= 50.0
    if service == "Power":
        return 20.0 <= value <= 25.0
    return False

# ------- Recommendation logic ------- #
def compute_recommendation(is_malicious_recommender: str | None, provider_is_benign_now: bool, provider_mal_type: str | None) -> bool:
    """
    Restituisce True -> "Provider is trustworthy", False -> "Provider is untrustworthy".
    - None (onesto): segue la verità dell'interazione corrente (provider_is_benign_now).
    - BMA: sempre 'untrustworthy'.
    - BSA: 'trustworthy' se il provider è malevolo in questa interazione, altrimenti 'untrustworthy'.
    """
    if is_malicious_recommender is None:
        return provider_is_benign_now
    if is_malicious_recommender == "BMA":
        return False
    if is_malicious_recommender == "BSA":
        return not provider_is_benign_now
    return provider_is_benign_now


def render_recommendations(requester: int, selected: int, provider_mal_type: str | None, last_eval_ok: bool):
    """Mostra le reccomandations dei neighbor del requester in base al risultato corrente."""
    if S.Gdt is None:
        st.info("No neighbors available for recommendations.")
        return
    try:
        neighbors = sorted(list(S.Gdt.neighbors(requester)))
    except Exception:
        neighbors = []
    # Escludi eventualmente il provider se è vicino
    neighbors = [n for n in neighbors if n != selected]

    if not neighbors:
        st.info("No neighbors available for recommendations.")
        return

    for n in neighbors:
        rec_type = S.rec_malicious_type.get(n)
        label = f"DT {n}" + (f" ({rec_type})" if rec_type else "")
        trusts = compute_recommendation(rec_type, last_eval_ok, provider_mal_type)
        icon = "✅" if trusts else "❌"
        verdict = "Provider is trustworthy" if trusts else "Provider is untrustworthy"
        st.write(f"{icon} {label}: {verdict}")



        
# -------------------- Init una tantum -------------------- #
def ensure_network_ready():
    if not S.created:
        S.seed = int(np.random.default_rng().integers(0, 10**9))
        S.Gdt, S.pos_rl3d, S.pos_dt3d = build_layers_3d(n=S.n, seed=S.seed)

        # Providers fissi per servizio (30–50%)
        rng = np.random.default_rng(S.seed + 100)
        nodes = np.arange(S.n)
        for srv in SERVICES:
            m_min = int(np.ceil(0.30 * S.n))
            m_max = int(np.floor(0.50 * S.n))
            m = int(rng.integers(m_min, m_max + 1))
            prov = set(map(int, rng.choice(nodes, size=m, replace=False)))
            S.providers_by_service[srv] = prov

        # --------- Malevoli sul servizio: 50% (ME/DA/OOA) ---------
        rng_m = np.random.default_rng(S.seed + 2025)
        num_mal = S.n // 2
        mal_nodes = rng_m.choice(nodes, size=num_mal, replace=False).tolist()
        k = num_mal // 3
        r = num_mal - 2 * k
        me_nodes = set(mal_nodes[:k])
        da_nodes = set(mal_nodes[k:2*k])
        ooa_nodes = set(mal_nodes[2*k:2*k + r])
        S.malicious_type = {}
        for i in nodes:
            if i in me_nodes:
                S.malicious_type[int(i)] = "ME"
            elif i in da_nodes:
                S.malicious_type[int(i)] = "DA"
            elif i in ooa_nodes:
                S.malicious_type[int(i)] = "OOA"

        # --------- Malevoli nelle raccomandazioni: 50% (BMA/BSA) separati ---------
        rng_r = np.random.default_rng(S.seed + 3030)
        num_rec_mal = S.n // 2
        rec_mal_nodes = rng_r.choice(nodes, size=num_rec_mal, replace=False).tolist()
        half = num_rec_mal // 2
        bma_nodes = set(rec_mal_nodes[:half])
        bsa_nodes = set(rec_mal_nodes[half:])
        S.rec_malicious_type = {}
        for i in nodes:
            if i in bma_nodes:
                S.rec_malicious_type[int(i)] = "BMA"
            elif i in bsa_nodes:
                S.rec_malicious_type[int(i)] = "BSA"
            else:
                S.rec_malicious_type[int(i)] = None

        S.created = True

ensure_network_ready()

# -------------------- Layout -------------------- #
left, right = st.columns([1.2, 1])

with left:
    st.title("Digital Twins Network")
    graph_placeholder = st.empty()

def draw_graph():
    fig = fig_two_layers_3d(S.Gdt, S.pos_rl3d, S.pos_dt3d, S.requester, S.providers, S.selected)
    graph_placeholder.plotly_chart(fig, use_container_width=True, key=GRAPH_KEY, config={"displayModeBar": False})

with right:
    st.subheader("Service Evaluation Scenario")

    # Requester & Service
    S.requester = st.selectbox("Requester", list(range(S.n)), index=min(S.requester, S.n-1), format_func=lambda i: f"DT {i}", key=f"{PAGE_TAG}-req")
    S.service = st.selectbox("Service", SERVICES, index=SERVICES.index(S.service) if S.service in SERVICES else 0, key=f"{PAGE_TAG}-srv")

    # Provider set (escludi requester)
    base_set = set(S.providers_by_service.get(S.service, set()))
    base_set.discard(S.requester)
    new_providers = sorted(base_set)
    if new_providers != S.providers:
        S.providers = new_providers
        S.selected = None
        S.evaluated = False
        S.last_eval_ok = None

    # Helper per label con tipo malevolo (sul servizio)
    def provider_label(i: int) -> str:
        t = S.malicious_type.get(i)
        return f"DT {i} ({t})" if t else f"DT {i}"

    if S.providers:
        st.markdown("**Service Discovery**")
        sel = st.radio(
            "Providers",
            options=S.providers,
            format_func=provider_label,
            horizontal=True,
            label_visibility="collapsed",
            key=f"{PAGE_TAG}-providers-radio-{S.service}-{S.requester}"
        )
        if sel != S.selected:
            S.selected = sel
            S.evaluated = False
            S.last_eval_ok = None

        # Mostra anteprima (non altera contatori OOA)
        if S.selected is not None:
            mal_t = S.malicious_type.get(S.selected)
            ooa_cnt = S.ooa_counter.get((S.requester, S.selected), 0)
            rng_preview = np.random.default_rng(S.seed + 1234 + S.selected + 100 * ooa_cnt)
            preview_val = provider_value(S.service, S.requester, S.selected, mal_t, ooa_cnt, rng_preview)
            S.readings[(S.service, S.selected)] = preview_val
            unit = " °C" if S.service == "Temperature" else (" %" if S.service == "Humidity" else " W")
            st.metric(f"{S.service} from {provider_label(S.selected)}", f"{preview_val:.1f}{unit}")

            # --- Due pulsanti affiancati: Reccomandations (popover) a sinistra + Evaluate Service a destra ---
            c_rec, c_eval = st.columns([0.5, 1.5])

            # 1) POPUP RECcomandations: sempre disponibile.
            with c_rec.popover("Reccomandations"):
                # Se non esiste ancora una valutazione "reale", usiamo la preview corrente
                provider_is_benign_for_recs = S.last_eval_ok
                if provider_is_benign_for_recs is None:
                    provider_is_benign_for_recs = is_value_benign(S.service, preview_val)
                render_recommendations(
                    requester=S.requester,
                    selected=S.selected,
                    provider_mal_type=mal_t,
                    last_eval_ok=provider_is_benign_for_recs,
                )

            # 2) EVALUATE SERVICE: esegue la valutazione "reale" (avanza OOA, salva esito)
            eval_btn_key = f"{PAGE_TAG}-eval-btn-{S.service}-{S.requester}-{S.selected}"
            with c_eval:
                if st.button("Evaluate Service", key=eval_btn_key):
                    curr_cnt = S.ooa_counter.get((S.requester, S.selected), 0)
                    rng_eval = np.random.default_rng(S.seed + 777 + S.selected + 1000 * curr_cnt)
                    eval_val = provider_value(S.service, S.requester, S.selected, mal_t, curr_cnt, rng_eval)
                    S.readings[(S.service, S.selected)] = eval_val
                    # aggiorna contatore se OOA
                    if mal_t == "OOA":
                        S.ooa_counter[(S.requester, S.selected)] = curr_cnt + 1
                    # esito ufficiale su soglie benevole
                    S.last_eval_ok = is_value_benign(S.service, eval_val)
                    S.evaluated = True

            # --- Verdettto del servizio (sempre sotto ai due pulsanti) ---
            if S.evaluated:
                if S.last_eval_ok:
                    st.success("Correct")
                else:
                    st.error("Wrong")


draw_graph()
