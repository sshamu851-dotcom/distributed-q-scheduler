import streamlit as st
import numpy as np
import pandas as pd
import time
from q_agent import QAgent
from utils import now_ms

# --- Page setup ---
st.set_page_config(page_title="Distributed Task Scheduler (Q-Learning)", layout="wide")

st.title("🧠 Distributed Task Scheduler using Q-Learning")
st.caption("A reinforcement-learning demo that learns to assign tasks efficiently among multiple workers.")

# --- Parameters ---
NUM_WORKERS = 3
if "agent" not in st.session_state:
    st.session_state.agent = QAgent(num_workers=NUM_WORKERS)
    st.session_state.worker_queues = [0] * NUM_WORKERS
    st.session_state.task_log = []
    st.session_state.epsilon_history = []
    st.session_state.avg_time_history = []

agent = st.session_state.agent
worker_queues = st.session_state.worker_queues
task_log = st.session_state.task_log

# --- User input ---
col1, col2 = st.columns(2)
complexity = col1.slider("Task Complexity (seconds)", 0.1, 3.0, 1.0, 0.1)
submit = col2.button("Submit Task")

# --- Submit task logic ---
if submit:
    state = agent.state_from_queues(worker_queues)
    chosen = agent.choose_worker(state)
    created = now_ms()
    processing_time = complexity * np.random.uniform(0.8, 1.2)
    worker_queues[chosen] += 1
    time.sleep(processing_time)          # simulate execution
    worker_queues[chosen] -= 1
    finished = now_ms()
    elapsed = (finished - created) / 1000.0
    reward = -elapsed
    next_state = agent.state_from_queues(worker_queues)
    agent.update(state, chosen, reward, next_state)

    task_log.append({
        "Worker": chosen,
        "Complexity": round(complexity, 2),
        "Time (s)": round(elapsed, 2),
        "Epsilon": round(agent.epsilon, 3)
    })

    # Update graphs
    st.session_state.epsilon_history.append(agent.epsilon)
    st.session_state.avg_time_history.append(elapsed)

# --- Dashboard display ---
st.subheader("⚙ Worker Queues")
cols = st.columns(NUM_WORKERS)
for i in range(NUM_WORKERS):
    cols[i].metric(f"Worker {i}", f"{worker_queues[i]} tasks")

st.divider()

st.subheader("📊 Learning Progress")
colA, colB = st.columns(2)
if st.session_state.epsilon_history:
    colA.line_chart(st.session_state.epsilon_history, y_label="Epsilon")
if st.session_state.avg_time_history:
    colB.line_chart(st.session_state.avg_time_history, y_label="Task Time (s)")

st.divider()

st.subheader("🧾 Task Log")
if task_log:
    st.dataframe(pd.DataFrame(task_log)[::-1])
else:
    st.info("No tasks yet — use the slider and click *Submit Task*.")

st.divider()

st.subheader("📚 Q-Table Snapshot")
if agent.Q:
    qdf = pd.DataFrame([
        {"State": str(k), **{f"W{i}": round(v, 3) for i, v in enumerate(vals)}}
        for k, vals in list(agent.Q.items())[-8:]
    ])
    st.dataframe(qdf)
else:
    st.info("Q-table is empty. Submit some tasks to start learning!")
