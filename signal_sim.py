"""
SmartSignal - Q-learning traffic light for one 4-way junction (Committee Chowk, Rawalpindi).

Roads:  North-South = Murree Road (busy),  East-West = side road (lighter).
Every 5 seconds the controller chooses:  0 = keep current green,  1 = switch.
Switching costs one 5-second yellow/all-red step (no car moves).
"""
import numpy as np

STEP_SEC = 5
EPISODE_STEPS = 360           # 30 minutes
SAT_FLOW = 2                  # max cars leaving one approach per green step (2 cars / 5 s)
QUEUE_BINS = [0, 3, 6, 10, 15, 22]     # -> 7 levels (0..6)
N_Q = len(QUEUE_BINS) + 1
N_PHASE, N_TIMER = 2, 3
N_STATES = N_Q * N_Q * N_PHASE * N_TIMER   # 7*7*2*3 = 294
N_ACTIONS = 2
BIG_QUEUE, BIG_QUEUE_PENALTY = 25, 5.0     # stop one road being starved


def qlevel(q):
    return int(np.searchsorted(QUEUE_BINS, q, side="right"))


class Junction:
    """Arrival rates are cars per 5-second step on each approach (N, S, E, W)."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, rates=None, seed=None):
        if rates is None:                          # random traffic level each episode
            rush = self.rng.uniform(0.6, 1.3)
            rates = np.array([0.55, 0.50, 0.30, 0.25]) * rush
        self.rates = np.asarray(rates, dtype=float)
        # pre-sample arrivals; with the same seed every controller faces exactly the same cars
        arr_rng = self.rng if seed is None else np.random.default_rng(seed)
        self.arrivals = arr_rng.poisson(self.rates, size=(EPISODE_STEPS, 4))
        self.q = np.zeros(4, dtype=int)
        self.phase, self.timer, self.yellow, self.t = 0, 0, 0, 0
        self.stats = dict(arrived=0, served=0, queue_sum=0.0, max_queue=0, switches=0)
        return self.state()

    def state(self):
        ns, ew = self.q[0] + self.q[1], self.q[2] + self.q[3]
        timer = 0 if self.timer < 3 else (1 if self.timer < 6 else 2)
        return ((qlevel(ns) * N_Q + qlevel(ew)) * N_PHASE + self.phase) * N_TIMER + timer

    def step(self, action):
        st = self.stats
        if action == 1 and self.yellow == 0:
            self.yellow = 1                        # next step is yellow, then the other road gets green
            st["switches"] += 1
        served = 0
        if self.yellow:
            self.yellow = 0
            self.phase = 1 - self.phase
            self.timer = 0
        else:
            lanes = (0, 1) if self.phase == 0 else (2, 3)
            for l in lanes:
                m = min(self.q[l], SAT_FLOW)
                self.q[l] -= m
                served += m
            self.timer += 1
        a = self.arrivals[self.t]
        self.q += a
        st["arrived"] += int(a.sum())
        st["served"] += served
        total = int(self.q.sum())
        st["queue_sum"] += total
        st["max_queue"] = max(st["max_queue"], int(self.q.max()))
        # reward: fewer waiting cars is better; extra penalty if any road's queue is very long
        r = -total / 10.0
        if self.q.max() > BIG_QUEUE:
            r -= BIG_QUEUE_PENALTY
        self.t += 1
        return self.state(), r, self.t >= EPISODE_STEPS

    def summary(self):
        st = self.stats
        return {
            # Little's law: total car-seconds waiting / cars that arrived
            "avg_wait_sec": st["queue_sum"] * STEP_SEC / max(st["arrived"], 1),
            "cars_served": st["served"],
            "avg_queue": st["queue_sum"] / EPISODE_STEPS,
            "max_queue": st["max_queue"],
            "switches": st["switches"],
        }


# ------------------------------------------------------------------ controllers
class FixedTime:
    """Today's signal: 30 s green for each road, no matter how many cars there are."""
    name = "Fixed-time (30 s)"

    def __init__(self, green_steps=6):
        self.g = green_steps

    def act(self, env, s):
        return 1 if env.timer >= self.g else 0


class RandomCtrl:
    name = "Random"

    def act(self, env, s):
        return int(env.rng.random() < 0.2)


class LongestQueue:
    """Rule: after a minimum green of 10 s, give green to the road with the longer queue."""
    name = "Longest-queue rule"

    def act(self, env, s):
        ns, ew = env.q[0] + env.q[1], env.q[2] + env.q[3]
        mine, other = (ns, ew) if env.phase == 0 else (ew, ns)
        return 1 if env.timer >= 2 and other > mine + 2 else 0


class QAgent:
    name = "Q-learning"

    def __init__(self, alpha=0.1, gamma=0.95, eps=1.0, eps_min=0.05, eps_decay=0.99, seed=0):
        self.Q = np.zeros((N_STATES, N_ACTIONS))
        self.N = np.zeros((N_STATES, N_ACTIONS), dtype=int)     # visit counts
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
        self.rng = np.random.default_rng(seed)
        self.training = True

    def act(self, env, s):
        if self.training and self.rng.random() < self.eps:
            return int(self.rng.integers(N_ACTIONS))           # explore
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))   # exploit

    def update(self, s, a, r, s2, done):
        target = r if done else r + self.gamma * self.Q[s2].max()
        self.N[s, a] += 1
        alpha = max(0.01, min(self.alpha, 1.0 / self.N[s, a] ** 0.5))   # shrinks as (s,a) is seen more
        self.Q[s, a] += alpha * (target - self.Q[s, a])

    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)
        self.training = False
        return self


def run(env, ctrl, rates=None, learn=False, seed=None):
    s = env.reset(rates, seed)
    total, done = 0.0, False
    while not done:
        a = ctrl.act(env, s)
        s2, r, done = env.step(a)
        if learn:
            ctrl.update(s, a, r, s2, done)
        total += r
        s = s2
    out = env.summary()
    out["reward"] = total
    return out


def decode(s):
    timer = s % N_TIMER; s //= N_TIMER
    phase = s % N_PHASE; s //= N_PHASE
    return s // N_Q, s % N_Q, phase, timer
