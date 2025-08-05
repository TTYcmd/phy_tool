import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from io import BytesIO
from PIL import Image
import tempfile
import os

st.set_page_config(layout="wide")
st.title("🎯 受迫阻尼振动教学演示")

# --- Sidebar controls ---
st.sidebar.header("参数调节")
m = st.sidebar.slider("质量 m (kg)", 0.1, 10.0, 1.0, step=1.0)
k = st.sidebar.slider("弹簧常数 k (N/m)", 0.1, 20.0, 10.0, step=1.0)
b = st.sidebar.slider("阻尼系数 b (N·s/m)", 0.0, 20.0, 1.0, step=1.0)
F0 = st.sidebar.slider("驱动力幅值 F₀ (N)", 0.0, 10.0, 0.0, step=1.0)
omega = st.sidebar.slider("驱动角频率 ω (rad/s)", 0.0, 20.0, 0.0, step=1.0)

# --- Initial condition ---
x0 = 1.0
v0 = 0.0

# --- Derived quantities ---
omega0 = np.sqrt(k / m)
zeta = b / (2 * np.sqrt(k * m))

# --- Display computed values ---
st.subheader("🔎 实时推导值")
col1, col2 = st.columns(2)
col1.metric("固有频率 ω₀ (rad/s)", f"{omega0:.3f}")
col2.metric("阻尼比 ζ", f"{zeta:.3f}")

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
        st.error("⚠️ 驱动频率接近固有频率，可能发生共振！")
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

# --- Check for solution validity ---
if sol.status != 0 or sol.y.shape[1] != len(sol.t):
    st.error("⚠️ 解算失败：请检查参数设置是否合理。")
    st.stop()


# --- Ball animation ---
st.subheader("🎬 小球受迫振动动画")

# 创建帧序列
frames = []
fig2, ax2 = plt.subplots(figsize=(8, 2))
ax2.set_xlim(-2, 2)
ax2.set_ylim(-1, 1)
ax2.set_aspect('equal')
ax2.axis('off')


for i in range(0, len(sol.t), 5):  # 每5帧取1，减少帧数，加快渲染
    ax2.clear()
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-1, 1)
    ax2.axis('off')
    x = sol.y[0][i]
    ax2.plot(x, 0, 'ro', markersize=20)
    ax2.plot(0, 0, 'ko', markersize=6)    # 中心黑点，坐标原点

    
    # 保存当前帧为图像
    buf = BytesIO()
    fig2.savefig(buf, format='png')
    buf.seek(0)
    frame = Image.open(buf).convert("RGB")
    frames.append(frame)
    buf.close()

plt.close(fig2)  # 避免显示空白图

# 保存为 GIF 动画
with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
    gif_path = tmpfile.name

frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=50,  # 每帧持续时间(ms)
    loop=0
)

# 读取 gif 数据
with open(gif_path, "rb") as f:
    gif_bytes = f.read()
    st.image(gif_bytes, caption="小球受迫振动位移动画", use_container_width=True)

# 可选：删除临时文件
try:
    os.remove(gif_path)
except Exception as e:
    print("Warning: 临时文件未能删除 ->", e)


# --- Plot x(t) ---
st.subheader("📈 位移-时间图像")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(sol.t, sol.y[0], label="Displacement x(t)", color="blue")
ax.set_xlabel("Time t (s)")
ax.set_ylabel("Displacement x(t) (m)")
ax.set_title("Displacement-Time Graph")
ax.grid(True)
ax.legend()
st.pyplot(fig)