import streamlit as st
import numpy as np
import pandas as pd
import time
from q_agent import QAgent
from utils import now_ms

st.set_page_config(page_title="Distributed Task Scheduler with Q-Learning", layout="wide")

st.title("🧠 Distributed Task Scheduler using Q-Learning")
st.caption("Simulated in Streamlit — demonstrates how a Q-learning agent distributes tasks among workers efficiently.")

NUM_WORKERS = 3

if "agent" not in st.session_state:
    st.session_state.agent = QAgent(num_workers=NUM_WORKERS)
    st.session_state.worker_queues = [0] * NUM_WORKERS
    st.session_state.task_log = []

agent = st.session_state.agent
worker_queues = st.session_state.worker_queues
task_log = st.session_state.task_log

col1, col2 = st.columns(2)
complexity = col1.slider("Task Complexity (seconds)", 0.1, 3.0, 1.0, 0.1)
submit = col2.button("Submit Task")

if submit:
    state = agent.state_from_queues(worker_queues)
    chosen = agent.choose_worker(state)
    created = now_ms()
    processing_time = complexity * np.random.uniform(0.8, 1.2)
    worker_queues[chosen] += 1
    time.sleep(processing_time)
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

st.subheader("⚙️ Worker Status")
cols = st.columns(NUM_WORKERS)
for i in range(NUM_WORKERS):
    cols[i].metric(f"Worker {i}", f"{worker_queues[i]} tasks")

st.subheader("📊 Q-Table Snapshot")
if agent.Q:
    st.write(f"Epsilon: {agent.epsilon:.4f}")
    qdf = pd.DataFrame([{ "State": str(k), **{f"W{i}": round(v,3) for i,v in enumerate(vals)} } for k, vals in list(agent.Q.items())[-5:]])
    st.dataframe(qdf)
else:
    st.info("No learning yet — submit tasks to start!")

st.subheader("🧾 Task Log")
if task_log:
    df = pd.DataFrame(task_log)
    st.dataframe(df[::-1])
else:
    st.info("No tasks yet — click 'Submit Task' above.")
