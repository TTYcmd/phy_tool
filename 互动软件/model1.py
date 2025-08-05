import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(layout="wide")
st.title("🎯 受迫阻尼振动教学演示")

# --- Sidebar controls ---
st.sidebar.header("参数调节")

m = st.sidebar.slider("质量 m (kg)", 0.1, 10.0, 1.0, step=1)
k = st.sidebar.slider("弹簧常数 k (N/m)", 0.1, 50.0, 10.0, step=1)
b = st.sidebar.slider("阻尼系数 b (N·s/m)", 0.0, 20.0, 1.0, step=1)
F0 = st.sidebar.slider("驱动力幅值 F₀ (N)", 0.0, 10.0, 1.0, step=1)
omega = st.sidebar.slider("驱动角频率 ω (rad/s)", 0.0, 20.0, 5.0, step=1)

# Initial condition
x0 = 1.0  # initial displacement
v0 = 0.0  # initial velocity

# --- Compute derived quantities ---
omega0 = np.sqrt(k / m)  # natural frequency
zeta = b / (2 * np.sqrt(k * m))  # damping ratio

# --- Display computed values ---
st.subheader("🔎 Real-time Derived Values")
col1, col2 = st.columns(2)
col1.metric("固有頻率 ω₀ (rad/s)", f"{omega0:.3f}")
col2.metric("阻尼係數 ζ", f"{zeta:.3f}")

# --- Classify motion ---
if F0 == 0:
    if zeta < 1:
        st.success("当前为：**欠阻尼自由振动**")
    elif np.isclose(zeta, 1.0, atol=0.01):
        st.info("当前为：**临界阻尼自由振动**")
    else:
        st.warning("当前为：**过阻尼自由振动**")
else:
    if abs(omega - omega0) < 0.1:
        st.error("⚠️ 驱动频率接近自然频率，共振可能发生！")
    else:
        st.info("当前为：**受迫阻尼振动**")

# --- Define ODE system ---
def damped_forced_osc(t, y):
    x, v = y
    dxdt = v
    dvdt = (-b * v - k * x + F0 * np.cos(omega * t)) / m
    return [dxdt, dvdt]

# --- Solve ODE ---
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)
sol = solve_ivp(damped_forced_osc, t_span, [x0, v0], t_eval=t_eval)

# --- Plot x(t) ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(sol.t, sol.y[0], label="Displacement x(t)", color="blue")
ax.set_xlabel("Time t (s)")
ax.set_ylabel("Displacement x(t) (m)")
ax.set_title("Displacement-Time Graph")
ax.grid(True)
ax.legend()

st.pyplot(fig)
