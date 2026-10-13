"""Builds the lab-task report (DOCX).   python build_report.py   (optional: set DEPLOY_URL)"""
import json
import os

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(os.path.dirname(HERE), "SmartSignal_RL_Lab_Task.docx")
URL = os.environ.get("DEPLOY_URL", "https://<your-app>.streamlit.app   (paste your link here)")

J = json.load(open(os.path.join(RES, "summary.json")))
S = J["summary"]
FX, RD, LQ, QL = "Fixed-time (30 s)", "Random", "Longest-queue rule", "Q-learning"
lvl = pd.read_csv(os.path.join(RES, "wait_by_traffic.csv"), index_col=0)
cut = 100 * (S[FX]["avg_wait_sec"] - S[QL]["avg_wait_sec"]) / S[FX]["avg_wait_sec"]

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11)
for h in ("Heading 1", "Heading 2"):
    doc.styles[h].font.color.rgb = RGBColor(0x0D, 0x47, 0xA1)


def para(t, bold=False, italic=False, size=None, center=False, after=6):
    p = doc.add_paragraph(); r = p.add_run(t); r.bold, r.italic = bold, italic
    if size: r.font.size = Pt(size)
    if center: p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(after)


def bullets(items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        if isinstance(it, tuple):
            p.add_run(it[0]).bold = True; p.add_run(it[1])
        else:
            p.add_run(it)


def table(head, rows, widths=None, fs=9.5):
    t = doc.add_table(rows=1, cols=len(head)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(head):
        c = t.rows[0].cells[i]; c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(fs)
        sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "BBDEFB"); c._tc.get_or_add_tcPr().append(sh)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(fs)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths): row.cells[i].width = Inches(w)
    doc.add_paragraph()


def fig(f, cap, w=6.2):
    doc.add_picture(os.path.join(RES, f), width=Inches(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(cap, italic=True, size=9, center=True)


# ---------------------------------------------------------------- cover
para("PMAS-Arid Agriculture University, Rawalpindi", bold=True, size=16, center=True)
para("University Institute of Information Technology (UIIT)", size=12, center=True)
para("Reinforcement Learning (CAI-461) – Lab / Practice Task", size=12, center=True, after=30)
para("SmartSignal", bold=True, size=24, center=True)
para("A Q-Learning Traffic Light for Committee Chowk, Rawalpindi", italic=True, size=14, center=True, after=30)
table(["Field", "Details"], [["Submitted To", "Dr. Sarfraz Bibi"], ["Submitted By", ""], ["Registration No.", ""],
                             ["Semester / Section", ""], ["Date", ""], ["Live MVP link", URL]], widths=[2.0, 4.3], fs=11)
doc.add_page_break()

doc.add_heading("Summary", 1)
para(f"Traffic lights at busy Rawalpindi junctions such as Committee Chowk run fixed timers: each road gets green for "
     f"the same time whether 2 or 20 cars are waiting. I modelled one junction as an MDP and trained a Q-learning "
     f"controller that decides every 5 seconds whether to keep or switch the green. In 40 test runs on identical "
     f"traffic it cut the average wait from {S[FX]['avg_wait_sec']:.1f} s to {S[QL]['avg_wait_sec']:.1f} s "
     f"(−{cut:.0f}%) and the longest queue from {S[FX]['max_queue']:.1f} to {S[QL]['max_queue']:.1f} cars, matching a "
     f"hand-designed 'longest-queue' rule without being told that rule. The MVP is deployed as a Streamlit app.")

# ---------------------------------------------------------------- step 1
doc.add_heading("Step 1 – Problem: adaptive traffic-signal control", 1)
para("Why this is a real RL problem: the controller acts repeatedly, each action changes the next situation (queues "
     "grow or shrink), rewards are delayed (switching now costs a yellow but helps later), and traffic is random. "
     "Adaptive and RL-based signal control is used and researched worldwide (Step 2).")
table(["MDP part", "Definition"], [
    ["Agent", "The signal controller of the junction."],
    ["Environment", "4-way junction: Murree Road (North–South, busier) and a side road (East–West). Cars arrive "
                    "randomly (Poisson), at most 2 cars per approach leave per 5 s of green."],
    ["State", "(Murree Rd queue level 0–6, side-road queue level 0–6, which road is green, time on green: "
              "<15 s / 15–30 s / >30 s) → 7×7×2×3 = 294 states."],
    ["Actions", "0 = keep current green, 1 = switch (costs one 5-second yellow during which no car moves)."],
    ["Transition", "Queues: q′ = q − departures(green lanes) + random arrivals."],
    ["Reward", "Every 5 s: r = −(total cars waiting)/10, and −5 extra if any approach has more than 25 cars."],
    ["Episode", "30 minutes = 360 decisions; each episode has a random traffic level (light → rush hour)."],
], widths=[1.4, 5.0])
doc.add_heading("Why this reward, and reward hacking", 2)
bullets([
    ("−(cars waiting): ", "by Little's law, total waiting cars over time = total delay, so minimising it minimises "
                          "average wait. It is given every step, so learning is fast."),
    ("Long-queue penalty: ", "stops the agent from keeping Murree Road green forever and starving the side road."),
    ("Bad reward 1 – cars served only: ", "the agent keeps the busy road green forever (more cars per minute) and "
                                          "the side road never moves – unfair and unsafe."),
    ("Bad reward 2 – +1 per switch / no yellow cost: ", "the light flickers every 5 s; lost yellow time causes huge queues."),
    ("Bad reward 3 – penalty only for queue on the green road: ", "the agent learns to switch away from long queues "
                                                                  "to 'hide' them on red."),
])

# ---------------------------------------------------------------- step 2
doc.add_heading("Step 2 – Research (prepared with the help of AI, facts checked)", 1)
doc.add_heading("Case 1 – Surtrac, Pittsburgh (USA)", 2)
para("Surtrac, developed at Carnegie Mellon University, makes every intersection plan its own green schedule every "
     "second from live sensor data and share predicted traffic with neighbouring intersections. In the 2013 pilot on "
     "nine intersections in East Liberty, travel time fell by about 25% and waiting time by about 40% on average "
     "(results varied by time of day). Note: Surtrac is real-time schedule optimisation, not RL – but it is the "
     "best-known proof that adaptive signals beat fixed timers. Challenges: cameras/radar at every junction, "
     "coordination between junctions, cost.")
doc.add_heading("Case 2 – IntelliLight (deep RL, KDD 2018)", 2)
para("Wei, Zheng, Yao & Li (Penn State) trained a Deep Q-Network traffic-light controller and, unlike earlier "
     "work, tested it on a large real traffic dataset collected from surveillance cameras, and explained the learned "
     "policies with case studies. Challenges they report: earlier RL studies used only synthetic traffic, and "
     "learned policies are hard to interpret – so they analysed what the network learned. Many later RL methods "
     "(PressLight, CoLight) build on this line of work.")
doc.add_heading("How Rawalpindi is different", 2)
table(["Issue", "Developed-country systems", "Committee Chowk / Rawalpindi"], [
    ["Traffic mix", "Mostly cars, lane discipline", "Motorbikes, rickshaws, wagons, carts – queues are not "
                                                     "neat lines; detection needs camera-based counting."],
    ["Sensors", "Loop detectors / radar everywhere", "Few sensors – start with one camera per approach."],
    ["Power", "Reliable", "Load-shedding can switch signals off – needs UPS and a fixed-time fallback."],
    ["Human control", "Rare", "Traffic police often override signals manually at rush hour."],
    ["Data", "Years of counts", "Little data – a simulator + small Q-table is the realistic start."],
], widths=[1.2, 2.2, 3.0], fs=9)

# ---------------------------------------------------------------- step 3
doc.add_heading("Step 3 – Q-Learning", 1)
para("Q(s,a) ← Q(s,a) + α [ r + γ · max_a′ Q(s′,a′) − Q(s,a) ]", bold=True, center=True)
doc.add_heading("Worked example", 2)
para("Murree Road is green for 35 s with 2 cars left; the side road has 12 cars. State s = (NS level 1, EW level 4, "
     "NS green, timer >30 s). Current Q(s, keep) = −6.0, Q(s, switch) = −5.0. The agent switches. During the "
     "yellow, 3 new cars arrive, so 17 cars wait → r = −17/10 = −1.7. In the next state the best value is "
     "max Q(s′,·) = −4.0. With α = 0.1, γ = 0.95:")
table(["Quantity", "Value"], [["TD target", "−1.7 + 0.95 × (−4.0) = −1.7 − 3.8 = −5.5"],
                              ["TD error", "−5.5 − (−5.0) = −0.5"],
                              ["New Q(s, switch)", "−5.0 + 0.1 × (−0.5) = −5.05"]], widths=[2.2, 4.2])
para("Switch (−5.05) is still better than keep (−6.0), so the agent keeps switching in this situation.")
doc.add_heading("Hyper-parameters", 2)
table(["Parameter", "Value", "Reason"], [
    ["Learning rate α", "min(0.5, 1/√n(s,a)), floor 0.01", "Big steps for new states, small steps later to average out random arrivals."],
    ["Discount γ", "0.95", "Horizon ≈ 20 steps ≈ 100 s – about one or two signal cycles."],
    ["Exploration", "ε-greedy, ε 1.0 → 0.05 (×0.997 per episode)", "Try both actions everywhere early, then exploit; ε = 0 when deployed."],
    ["Episodes", "3,000 × 360 steps", "≈ 1 million updates, ~2 minutes on a laptop."],
], widths=[1.3, 2.0, 3.1], fs=9)
doc.add_heading("Is a Q-table enough?", 2)
para(f"Yes, for this MVP. Grouping queue lengths into 7 levels gives only {J['n_states']} states × 2 actions; "
     f"{J['states_visited']} states were visited in training. For a whole corridor of 10 junctions (294^10 joint states), "
     "exact queue counts, or camera images as input, function approximation (a DQN as in IntelliLight) or one agent "
     "per junction would be needed.")

# ---------------------------------------------------------------- step 4
doc.add_heading("Step 4 – MVP design", 1)
table(["In the MVP", "Left for later"], [
    ["One junction, two phases (N–S / E–W)", "Turning phases, pedestrian phase, a corridor of junctions"],
    ["Simulated Poisson traffic, light → rush hour", "Real camera vehicle counts (YOLO)"],
    ["Tabular Q-learning, 294 states", "Deep RL, coordination between junctions"],
    ["Baselines: fixed-time, random, longest-queue rule", "Emergency-vehicle priority"],
    ["Streamlit demo + charts", "Hardware controller integration"],
], widths=[3.2, 3.2], fs=9.5)
para("Safe testing and fallback: run in shadow mode first (agent suggests, police/fixed timer decides), enforce a "
     "minimum green of 10 s and maximum of 90 s, and fall back to the fixed-time plan if the camera fails, power "
     "drops, or any queue exceeds a limit. Traffic police can always override.")

# ---------------------------------------------------------------- step 5
doc.add_heading("Step 5 – Develop and test", 1)
para("Python + NumPy, written from scratch: signal_sim.py (junction + controllers + Q-learning), "
     "train_and_evaluate.py (training, tests, charts), app.py (Streamlit). Every controller is tested on exactly the "
     "same cars (same random seed), 10 runs at each of 4 traffic levels.")
fig("reward_curve.png", "Reward curve: reward rises from random-level to the level of the best rule as ε decays")
para(f"Average reward rose from {J['first50']:.0f} (first 50 episodes) to {J['last50']:.0f} (last 50).")
table(["Controller", "Avg wait (s)", "Max queue", "Cars served", "Switches", "Reward"],
      [[n, f"{S[n]['avg_wait_sec']:.1f}", f"{S[n]['max_queue']:.1f}", f"{S[n]['cars_served']:.0f}",
        f"{S[n]['switches']:.0f}", f"{S[n]['reward']:.0f}"] for n in (FX, RD, LQ, QL)], fs=9.5)
fig("metrics.png", "Test metrics (40 runs)")
fig("wait_by_traffic.png", "Average wait at each traffic level", 5.5)
fig("policy_map.png", "What the agent learned: switch when Murree Rd is nearly empty and the side road is waiting", 4.0)
doc.add_heading("Discussion", 2)
bullets([
    f"Q-learning cut average wait by {cut:.0f}% vs the fixed timer used today, at every traffic level.",
    f"It performs about the same as the hand-written longest-queue rule ({S[QL]['avg_wait_sec']:.1f} s vs "
    f"{S[LQ]['avg_wait_sec']:.1f} s) and has the shortest longest-queue ({S[QL]['max_queue']:.1f} cars) – it discovered a "
    "sensible rule from reward alone.",
    f"It switches more often ({S[QL]['switches']:.0f} vs {S[LQ]['switches']:.0f} per 30 min); a small switching penalty "
    "would reduce this.",
    "Limitation: simulated traffic only; real Rawalpindi traffic (motorbikes, manual police control) must be tested in shadow mode first.",
])

# ---------------------------------------------------------------- step 6
doc.add_heading("Step 6 – Deployment", 1)
p = doc.add_paragraph(); p.add_run("Live app: ").bold = True; p.add_run(URL)
para("Deployed on Streamlit Community Cloud from GitHub. The user picks the traffic level and the side-road share, "
     "and the app simulates 30 minutes for all four controllers and shows wait times and live queue graphs.")
bullets([("Risks: ", "unfair treatment of the side road, camera privacy (number plates), wrong decisions if a "
                     "sensor fails."),
         ("Mitigation: ", "long-queue penalty and max-red limit, count vehicles only and do not store video, "
                          "fixed-time fallback and police override.")])

doc.add_heading("References", 1)
for r in ["Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.",
          "Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine Learning, 8, 279–292.",
          "Smith, S. F., Barlow, G., Xie, X.-F., & Rubinstein, Z. B. (2013). Smart urban signal networks: Initial application "
          "of the SURTRAC adaptive traffic signal control system. ICAPS 2013.",
          "Wei, H., Zheng, G., Yao, H., & Li, Z. (2018). IntelliLight: A reinforcement learning approach for intelligent "
          "traffic light control. KDD 2018."]:
    doc.add_paragraph(r, style="List Number")
doc.save(OUT)
print("saved", OUT)
