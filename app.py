"""SmartSignal - Q-learning traffic light demo.   Run:  streamlit run app.py"""
import os

import numpy as np
import pandas as pd
import streamlit as st

import signal_sim as S

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
st.set_page_config(page_title="SmartSignal RL", page_icon="🚦", layout="wide")


@st.cache_resource
def agent():
    return S.QAgent().load(os.path.join(RES, "q_table.npy"))


def trace(ctrl, rates, seed):
    """Run one 30-minute episode and record queues every step."""
    env = S.Junction(seed)
    s = env.reset(rates, seed)
    ns, ew, phase, done = [], [], [], False
    while not done:
        s, _, done = env.step(ctrl.act(env, s))
        ns.append(env.q[0] + env.q[1]); ew.append(env.q[2] + env.q[3]); phase.append(env.phase)
    return env.summary(), np.array(ns), np.array(ew), np.array(phase)


st.title("🚦 SmartSignal – Q-learning traffic light at Committee Chowk")
st.caption("RL lab task · one 4-way junction · Murree Road (N–S) vs side road (E–W) · decision every 5 s")

with st.sidebar:
    st.header("Traffic")
    level = st.slider("Traffic level (1.0 = normal busy hour)", 0.5, 1.4, 1.0, 0.1)
    side = st.slider("Side-road share (E–W) vs Murree Road", 0.2, 0.8, 0.35, 0.05)
    seed = st.number_input("Random seed", 0, 9999, 7)

total = 1.6 * level
rates = np.array([(1 - side) / 2, (1 - side) / 2, side / 2, side / 2]) * total
ctrls = [S.FixedTime(), S.RandomCtrl(), S.LongestQueue(), agent()]

tab1, tab2, tab3 = st.tabs(["Simulate 30 minutes", "Training results", "How it works"])
with tab1:
    res = {c.name: trace(c, rates, int(seed)) for c in ctrls}
    df = pd.DataFrame({n: r[0] for n, r in res.items()}).T
    df.columns = ["Avg wait (s)", "Cars served", "Avg queue", "Max queue", "Switches"]
    fixed, rl = df.loc["Fixed-time (30 s)", "Avg wait (s)"], df.loc["Q-learning", "Avg wait (s)"]
    c = st.columns(3)
    c[0].metric("Q-learning avg wait", f"{rl:.1f} s", f"{100 * (rl - fixed) / fixed:+.0f}% vs fixed-time", delta_color="inverse")
    c[1].metric("Fixed-time avg wait", f"{fixed:.1f} s")
    c[2].metric("Q-learning longest queue", f"{df.loc['Q-learning', 'Max queue']:.0f} cars")
    st.dataframe(df.style.format("{:.1f}"), width="stretch")
    pick = st.radio("Show queues for", list(res), index=3, horizontal=True)
    _, ns, ew, ph = res[pick]
    t = np.arange(len(ns)) * S.STEP_SEC / 60
    st.line_chart(pd.DataFrame({"Murree Rd (N–S) queue": ns, "Side road (E–W) queue": ew}, index=np.round(t, 2)))
    st.caption("x-axis: minutes. Each switch of green costs a 5-second yellow.")

with tab2:
    for f, cap in [("reward_curve.png", "Reward curve"), ("metrics.png", "Test metrics"),
                   ("wait_by_traffic.png", "Wait by traffic level"), ("policy_map.png", "What the agent learned")]:
        p = os.path.join(RES, f)
        if os.path.exists(p):
            st.image(p, caption=cap, width="stretch")

with tab3:
    st.markdown(f"""
**State** (= {S.N_STATES} states, a plain Q-table is enough): queue level on Murree Rd (7) × queue level on side road (7)
× which road is green (2) × how long it has been green (3).

**Actions:** keep the current green, or switch (costs one 5-s yellow).

**Reward every 5 s:** −(cars waiting)/10, and −5 extra if any approach has more than {S.BIG_QUEUE} cars (so the side road is never starved).

**Q-learning:** Q(s,a) ← Q(s,a) + α [r + γ·max Q(s′,a′) − Q(s,a)], γ = 0.95, ε-greedy 1.0 → 0.05, α shrinking with visits.
""")
