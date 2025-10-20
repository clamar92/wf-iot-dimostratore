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
    "Always Collaborate": "AC",
    "Always Defect (ME)": "AD",
    "On-Off Attack": "ONOFF",           # switch every 2 rounds
    "Opportunistic Attack": "OPP",      # attack after 3 positive rounds
}

@dataclass
class PayoffParams:
    Rr: int = 2
    Rp: int = 2
    S:  int = -1
    D:  int = 3

def payoff(req: str, prov: str, p: PayoffParams) -> tuple[int, int]:
    if req == A_TRUST and prov == P_COLLAB:
        return (p.Rr, p.Rp)
    elif req == A_TRUST and prov == P_DEFECT:
        return (p.S, p.D)
    else:
        return (0, 0)

def maybe_flip(action: str, prob: float, rng: np.random.Generator) -> str:
    return (P_DEFECT if action == P_COLLAB else P_COLLAB) if rng.random() < prob else action

@dataclass
class GameState:
    prov_strategy: str = "AC"
    round_idx: int = 0
    score_req: int = 0
    score_prov: int = 0
    opp_positive_streak: int = 0
    errors_on: bool = False
    error_prob: float = 0.20
    seed: int = 2025
    payoff: PayoffParams = field(default_factory=PayoffParams)
    last_message: str = ""
    last_provider_action: str | None = None  # "C" or "D"

    def reset_match(self):
        """Reset the match: scores, round, last message, streak."""
        self.score_req = 0
        self.score_prov = 0
        self.round_idx = 0
        self.opp_positive_streak = 0
        self.last_message = ""
        self.last_provider_action = None
        # NOTE: strategy, sliders (payoffs), and error toggle intentionally preserved

    def provider_move(self, rng: np.random.Generator) -> str:
        s = self.prov_strategy
        if s == "AC":
            action = P_COLLAB
        elif s == "AD":
            action = P_DEFECT
        elif s == "ONOFF":
            block = (self.round_idx // 2)
            action = P_COLLAB if (block % 2 == 0) else P_DEFECT
        elif s == "OPP":
            if self.opp_positive_streak >= 3:
                action = P_DEFECT
                self.opp_positive_streak = 0
            else:
                action = P_COLLAB
        else:
            action = P_COLLAB
        if self.errors_on:
            action = maybe_flip(action, self.error_prob, rng)
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
    st.caption("Requester chooses **Trust** or **No-Trust**. Provider follows the selected strategy (may include attacks).")

    c1, c2, c3 = st.columns(3)
    c1.metric("Round", S.round_idx)
    c2.metric("Requester score", S.score_req)
    c3.metric("Provider score", S.score_prov)

    rng = np.random.default_rng(S.seed + S.round_idx * 97)
    cT, cN = st.columns(2)

    def play_round(req_action: str):
        prov_action = S.provider_move(rng)
        pr, pp = payoff(req_action, prov_action, S.payoff)

        # Opportunistic streak
        if req_action == A_TRUST and prov_action == P_COLLAB:
            S.opp_positive_streak += 1
        elif prov_action == P_DEFECT:
            S.opp_positive_streak = 0

        S.score_req += pr
        S.score_prov += pp
        S.round_idx += 1

        S.last_provider_action = prov_action
        S.last_message = (
            f"Last round → Requester: **{'Trust' if req_action=='T' else 'No-Trust'}**, "
            f"Provider: **{'Collaborate' if prov_action=='C' else 'Defect'}** → payoff: **{pr:+d}/{pp:+d}**"
        )
        st.rerun()  # ensure the updated message shows immediately

    if cT.button("TRUST", use_container_width=True, key=f"{PAGE_TAG}-T"):
        play_round(A_TRUST)
    if cN.button("NO-TRUST", use_container_width=True, key=f"{PAGE_TAG}-N"):
        play_round(A_NOTRUST)

    # Show last outcome (placed AFTER buttons so it reflects the current click)
    if S.last_message:
        (st.error if S.last_provider_action == P_DEFECT else st.success)(S.last_message)

    st.divider()
    if st.button("Reset match", use_container_width=True, key=f"{PAGE_TAG}-reset"):
        S.reset_match()
        st.rerun()

with right:
    st.subheader("Provider strategy")
    strat_label = st.radio(
        "Behaviour",
        options=list(PROV_STRATS.keys()),
        index=list(PROV_STRATS.values()).index(S.prov_strategy),
        horizontal=False,
        key=f"{PAGE_TAG}-prov-radio",
    )
    new_code = PROV_STRATS[strat_label]
    if new_code != S.prov_strategy:
        S.prov_strategy = new_code
        # keep current scores/round unless user resets

    st.divider()
    st.subheader("Payoffs")
    r1c1, r1c2 = st.columns(2)
    S.payoff.Rr = r1c1.slider("Rr (Requester)", -5, 10, S.payoff.Rr, 1)
    S.payoff.Rp = r1c2.slider("Rp (Provider)", -5, 10, S.payoff.Rp, 1)
    r2c1, r2c2 = st.columns(2)
    S.payoff.S  = r2c1.slider("S (Requester)", -5, 10, S.payoff.S, 1)
    S.payoff.D  = r2c2.slider("D (Provider)", -5, 10, S.payoff.D, 1)
    st.caption("Payoffs: Collaborate/Trust → [Rr, Rp]; Defect/Trust → [S, D]; otherwise [0, 0].")

    st.divider()
    st.subheader("Errors")
    S.errors_on = st.checkbox("Enable random errors (20%)", value=S.errors_on, key=f"{PAGE_TAG}-err")
