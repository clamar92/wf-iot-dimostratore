# app_step4.py — DT Request Demo — Two-player trust game (Trust vs No-Trust)
import streamlit as st
from dataclasses import dataclass, field
import numpy as np

PAGE_TAG = "step4"
st.set_page_config(page_title="DT Request Demo — Step 4", layout="wide")

# --- UI polish ---
st.markdown("""
<style>
header[data-testid="stHeader"] {display:none!important;}
.block-container{padding-top:0.5rem;}
html,body,[data-testid="stAppViewContainer"]{overflow-x:hidden!important;}
h1 a,h2 a,h3 a,h4 a,h5 a,h6 a,[data-testid="stMarkdownContainer"] h1 a,
[data-testid="stMarkdownContainer"] h2 a,[data-testid="stMarkdownContainer"] h3 a,
[data-testid="stMarkdownContainer"] h4 a,[data-testid="stMarkdownContainer"] h5 a,
[data-testid="stMarkdownContainer"] h6 a{pointer-events:none!important;color:inherit!important;text-decoration:none!important;}
</style>
""", unsafe_allow_html=True)

# -------------------- Model -------------------- #
A_TRUST, A_NOTRUST = "T", "N"  # Requester actions
P_COLLAB, P_DEFECT = "C", "D"  # Provider actions

PROV_STRATS = {
    "Behaviour 1": "B1",   # always cooperative + errors with prob p
    "Behaviour 2": "B2",   # always defect
    "Behaviour 3": "B3",   # on-off (switch every 2 rounds)
    "Behaviour 4": "B4",   # always cooperative + fixed 0.20 error
}

@dataclass
class PayoffParams:
    Rr: int = 2
    Rp: int = 2
    S:  int = -1
    D:  int = 3

def payoff(req: str, prov: str | None, p: PayoffParams) -> tuple[int, int]:
    if req == A_NOTRUST:
        return (0, 0)
    if prov == P_COLLAB:
        return (p.Rr, p.Rp)
    else:
        return (p.S, p.D)

def maybe_flip(action: str, prob: float, rng: np.random.Generator) -> str:
    return (P_DEFECT if action == P_COLLAB else P_COLLAB) if rng.random() < prob else action

@dataclass
class GameState:
    prov_strategy: str = "B1"
    round_idx: int = 0
    score_req: int = 0
    score_prov: int = 0
    error_prob: float = 0.0  # slider p
    seed: int = 2025
    payoff: PayoffParams = field(default_factory=PayoffParams)
    last_message: str = ""
    last_provider_action: str | None = None  # "C" / "D" / None

    def reset_match(self):
        self.score_req = 0
        self.score_prov = 0
        self.round_idx = 0
        self.last_message = ""
        self.last_provider_action = None

    def provider_move(self, rng: np.random.Generator) -> str:
        s = self.prov_strategy
        if s == "B1":
            action = P_COLLAB
            action = maybe_flip(action, self.error_prob, rng)  # errors only on Behaviour 1
        elif s == "B2":
            action = P_DEFECT
        elif s == "B3":
            block = (self.round_idx // 2)
            action = P_COLLAB if (block % 2 == 0) else P_DEFECT
        elif s == "B4":
            action = P_COLLAB
            action = maybe_flip(action, 0.20, rng)  # fixed 0.20
        else:
            action = P_COLLAB
        return action

# -------------------- State -------------------- #
STATE_KEY = f"{PAGE_TAG}-state"
if STATE_KEY not in st.session_state:
    st.session_state[STATE_KEY] = GameState()
S: GameState = st.session_state[STATE_KEY]

# -------------------- Layout -------------------- #
left, right = st.columns([1.1, 1])

with left:
    st.title("Step 4 · Trust Game (Requester vs Provider)")
    st.caption("Requester chooses **Trust** or **No-Trust**. Provider follows the selected behaviour.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Round", S.round_idx)
    c2.metric("Requester score", S.score_req)
    c3.metric("Provider score", S.score_prov)

    rng = np.random.default_rng(S.seed + S.round_idx * 97)
    cT, cN = st.columns(2)

    def play_round(req_action: str):
        if req_action == A_NOTRUST:
            pr, pp = payoff(req_action, None, S.payoff)
            S.score_req += pr; S.score_prov += pp
            S.round_idx += 1
            S.last_provider_action = None
            S.last_message = "Last round → Requester: **No-Trust** → payoff: **+0/+0**"
            st.rerun(); return

        prov_action = S.provider_move(rng)
        pr, pp = payoff(req_action, prov_action, S.payoff)
        S.score_req += pr; S.score_prov += pp
        S.round_idx += 1
        S.last_provider_action = prov_action
        S.last_message = (
            f"Last round → Requester: **Trust**; "
            f"Provider: **{'Collaborate' if prov_action=='C' else 'Defect'}** "
            f"→ payoff: **{pr:+d}/{pp:+d}**"
        )
        st.rerun()

    if cT.button("TRUST", use_container_width=True, key=f"{PAGE_TAG}-T"):
        play_round(A_TRUST)
    if cN.button("NO-TRUST", use_container_width=True, key=f"{PAGE_TAG}-N"):
        play_round(A_NOTRUST)

    if S.last_message:
        (st.error if S.last_provider_action == P_DEFECT else st.success)(S.last_message)

    st.divider()
    if st.button("Reset match", use_container_width=True, key=f"{PAGE_TAG}-reset"):
        S.reset_match()
        st.rerun()

with right:
    st.subheader("Provider behaviour")
    strat_label = st.radio(
        "Select behaviour",
        options=list(PROV_STRATS.keys()),
        index=list(PROV_STRATS.values()).index(S.prov_strategy) if S.prov_strategy in PROV_STRATS.values() else 0,
        horizontal=False,
        key=f"{PAGE_TAG}-prov-radio",
    )
    new_code = PROV_STRATS[strat_label]
    if new_code != S.prov_strategy:
        S.prov_strategy = new_code

    st.divider()
    st.subheader("Payoffs")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Rr (Requester)** = {S.payoff.Rr}")
        st.markdown(f"**S (Requester)** = {S.payoff.S}")
    with c2:
        st.markdown(f"**Rp (Provider)** = {S.payoff.Rp}")
        st.markdown(f"**D (Provider)** = {S.payoff.D}")


    st.divider()
    st.subheader("Errors")
    S.error_prob = st.slider("p: probability of errors", 0.0, 1.0, S.error_prob, 0.01)
