# SmartSignal – Q-learning traffic light (RL lab task)

One junction (Committee Chowk, Rawalpindi). Every 5 s a Q-learning agent decides: keep green or switch.
Compared with fixed-time (30 s), random, and a longest-queue rule.

```
pip install -r requirements.txt
python train_and_evaluate.py   # ~2 min, writes ./results
streamlit run app.py
python build_report.py         # writes ../SmartSignal_RL_Lab_Task.docx
```

Deploy: push this folder (with `results/`) to GitHub → share.streamlit.io → New app → main file `app.py`.
