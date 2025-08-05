import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(layout="wide")
st.title("🎯 受迫阻尼振动教学演示")

# --- UI 控件区 ---
st.sidebar.header("参数调节")

m = st.sidebar.slider("质量 m (kg)", 0.1, 10.0, 1.0, step=0.1)
k = st.sidebar.slider("弹簧常数 k (N/m)", 0.1, 50.0, 10.0, step=0.1)
b = st.sidebar.slider("阻尼系数 b (N·s/m)", 0.0, 20.0, 1.0, step=0.1)
F0 = st.sidebar.slider("驱动力幅值 F₀ (N)", 0.0, 10.0, 1.0, step=0.1)
omega = st.sidebar.slider("驱动角频率 ω (rad/s)", 0.0, 20.0, 5.0, step=0.1)

# 初始条件
x0 = 1.0  # 初始位移
v0 = 0.0  # 初始速度

# --- 计算自然频率和阻尼比 ---
omega0 = np.sqrt(k / m)
zeta = b / (2 * np.sqrt(k * m))

# --- 状态判断 ---
if F0 == 0:
    if zeta < 1:
        st.success("当前为：**欠阻尼自由振动**")
    elif np.isclose(zeta, 1):
        st.info("当前为：**临界阻尼自由振动**")
    else:
        st.warning("当前为：**过阻尼自由振动**")
else:
    if abs(omega - omega0) < 0.1:
        st.error("⚠️ 驱动频率接近自然频率，共振可能发生！")
    else:
        st.info("当前为：**受迫阻尼振动**")

# --- 微分方程定义 ---
def damped_forced_osc(t, y):
    x, v = y
    dxdt = v
    dvdt = (-b * v - k * x + F0 * np.cos(omega * t)) / m
    return [dxdt, dvdt]

# --- 数值求解 ---
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)
sol = solve_ivp(damped_forced_osc, t_span, [x0, v0], t_eval=t_eval)

# --- 画图 ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(sol.t, sol.y[0], label="Displacement x(t)", color="blue")
ax.set_xlabel("Time t (s)")
ax.set_ylabel("Displacement x(t) (m)")
ax.set_title("Displacement-Time Graph")
ax.grid(True)
ax.legend()

# --- 显示图像 ---
st.pyplot(fig)
