"""Train the Q-learning signal, compare with baselines, save charts.   python train_and_evaluate.py"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import signal_sim as S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
COL = {"Fixed-time (30 s)": "#9e9e9e", "Random": "#e57373", "Longest-queue rule": "#64b5f6", "Q-learning": "#2e7d32"}
BASE = np.array([0.55, 0.50, 0.30, 0.25])
LEVELS = {"Light (0.6x)": 0.6, "Normal (0.8x)": 0.8, "Busy (1.0x)": 1.0, "Rush hour (1.3x)": 1.3}

# ---------------- training
agent, env, log = S.QAgent(alpha=0.5, eps_decay=0.997), S.Junction(0), []
for ep in range(3000):
    r = S.run(env, agent, learn=True)
    log.append(dict(episode=ep + 1, reward=r["reward"], wait=r["avg_wait_sec"], eps=agent.eps))
    agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
agent.training = False
agent.save(os.path.join(OUT, "q_table.npy"))
log = pd.DataFrame(log); log.to_csv(os.path.join(OUT, "training_log.csv"), index=False)

# ---------------- evaluation: same cars for every controller (seeded), 10 runs per traffic level
ctrls = [S.FixedTime(), S.RandomCtrl(), S.LongestQueue(), agent]
rows = []
for lvl, k in LEVELS.items():
    for rep in range(10):
        for c in ctrls:
            r = S.run(S.Junction(rep), c, BASE * k, seed=1000 + rep)
            rows.append(dict(controller=c.name, traffic=lvl, **r))
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "eval_all.csv"), index=False)
order = [c.name for c in ctrls]
metrics = ["avg_wait_sec", "cars_served", "avg_queue", "max_queue", "switches", "reward"]
summ = df.groupby("controller")[metrics].mean().loc[order]
summ.round(2).to_csv(os.path.join(OUT, "summary.csv"))
by_lvl = df.pivot_table(index="traffic", columns="controller", values="avg_wait_sec").loc[list(LEVELS)][order]
by_lvl.round(2).to_csv(os.path.join(OUT, "wait_by_traffic.csv"))

# ---------------- charts
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(log.episode, log.reward, color="#a5d6a7", lw=0.6, label="per episode (exploring)")
ax.plot(log.episode, log.reward.rolling(50, min_periods=1).mean(), color="#2e7d32", lw=2.5, label="50-episode moving avg")
for n in ["Fixed-time (30 s)", "Longest-queue rule"]:
    ax.axhline(summ.loc[n, "reward"], ls="--", color=COL[n], label=f"{n} (test avg)")
ax2 = ax.twinx(); ax2.plot(log.episode, log.eps, color="#7e57c2", alpha=0.6); ax2.set_ylabel("epsilon", color="#7e57c2")
ax.set_xlabel("Episode (30 simulated minutes)"); ax.set_ylabel("Total reward"); ax.set_ylim(-900, 0)
ax.set_title("SmartSignal - Q-learning reward curve"); ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "reward_curve.png"), dpi=150); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (k, t) in zip(axes, [("avg_wait_sec", "Avg wait per car (s)"), ("max_queue", "Longest queue (cars)"),
                             ("cars_served", "Cars through junction / 30 min")]):
    v = summ[k]; ax.bar(range(4), v, color=[COL[n] for n in order])
    ax.set_xticks(range(4)); ax.set_xticklabels([n.replace(" (", "\n(").replace(" rule", "\nrule") for n in order], fontsize=8)
    for i, x in enumerate(v): ax.text(i, x, f"{x:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(t); ax.grid(axis="y", alpha=0.3)
    if k == "cars_served": ax.set_ylim(v.min() * 0.95, v.max() * 1.02)
fig.suptitle("Test results (40 runs, same cars for every controller)"); fig.tight_layout()
fig.savefig(os.path.join(OUT, "metrics.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(LEVELS)); w = 0.2
for i, n in enumerate(order):
    ax.bar(x + (i - 1.5) * w, by_lvl[n], w, color=COL[n], label=n)
ax.set_xticks(x); ax.set_xticklabels(list(LEVELS)); ax.set_ylabel("Avg wait (s)")
ax.set_title("Average wait by traffic level"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "wait_by_traffic.png"), dpi=150); plt.close(fig)

# learned policy map (NS green, after 30 s): green = keep, red = switch
pol = np.zeros((S.N_Q, S.N_Q))
for ns in range(S.N_Q):
    for ew in range(S.N_Q):
        s = ((ns * S.N_Q + ew) * S.N_PHASE + 0) * S.N_TIMER + 2
        pol[ns, ew] = np.nan if not agent.N[s].any() else float(agent.Q[s, 1] > agent.Q[s, 0])
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.imshow(pol, cmap="RdYlGn_r", vmin=0, vmax=1)
lab = ["0", "1-3", "4-6", "7-10", "11-15", "16-22", "23+"]
ax.set_xticks(range(7)); ax.set_xticklabels(lab); ax.set_yticks(range(7)); ax.set_yticklabels(lab)
ax.set_xlabel("Cars waiting on side road (E-W, red)"); ax.set_ylabel("Cars waiting on Murree Rd (N-S, green)")
for i in range(7):
    for j in range(7):
        ax.text(j, i, "" if np.isnan(pol[i, j]) else ("SWITCH" if pol[i, j] else "keep"), ha="center", va="center", fontsize=6.5)
ax.set_title("Learned policy (N-S green for 30 s+)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "policy_map.png"), dpi=150); plt.close(fig)

json.dump({"summary": summ.round(3).to_dict(orient="index"), "by_level": by_lvl.round(2).to_dict(),
           "states_visited": int((agent.N.sum(1) > 0).sum()), "n_states": S.N_STATES,
           "first50": float(log.reward.head(50).mean()), "last50": float(log.reward.tail(50).mean())},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
pd.set_option("display.width", 200)
print(summ.round(2)); print(by_lvl.round(2))
