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

# 预设参数字典
preset_params = {
    "欠阻尼": {"m":2.0, "k":2.0, "b":1.0, "F0":0.0, "omega":0.0},
    "临界阻尼": {"m":1.0, "k":1.0, "b":2*np.sqrt(1), "F0":0.0, "omega":0.0},
    "过阻尼": {"m":1.0, "k":1.0, "b":5.0, "F0":0.0, "omega":0.0},
    "共振": {"m":1.0, "k":1.0, "b":1.0, "F0":1.0, "omega":np.sqrt(1)},
    "自定义": None,
}

# 1. 选择状态
state = st.sidebar.radio("选择预设状态", list(preset_params.keys()))

use_preset = (state != "自定义")

# 2. 参数赋值（禁用滑块编辑预设参数）
if use_preset:
    params = preset_params[state]
    m = st.sidebar.slider("质量 m (kg)", 0.1, 10.0, value=params["m"], step=1.0, disabled=True)
    k = st.sidebar.slider("劲度系数 k (N/m)", 0.1, 20.0, value=params["k"], step=1.0, disabled=True)
    b = st.sidebar.slider("阻尼系数 b (N·s/m)", 0.0, 20.0, value=params["b"], step=1.0, disabled=True)
    F0 = st.sidebar.slider("驱动力幅值 F₀ (N)", 0.0, 10.0, value=params["F0"], step=1.0, disabled=True)
    omega = st.sidebar.slider("驱动角频率 ω (rad/s)", 0.0, 20.0, value=params["omega"], step=1.0, disabled=True)
else:
    m = st.sidebar.slider("质量 m (kg)", 0.1, 10.0, 1.0, step=1.0)
    k = st.sidebar.slider("劲度系数 k (N/m)", 0.1, 20.0, 1.0, step=1.0)
    b = st.sidebar.slider("阻尼系数 b (N·s/m)", 0.0, 20.0, 1.0, step=1.0)
    F0 = st.sidebar.slider("驱动力幅值 F₀ (N)", 0.0, 10.0, 1.0, step=1.0)
    omega = st.sidebar.slider("驱动角频率 ω (rad/s)", 0.0, 20.0, 3.0, step=1.0)

# 3. 计算并缓存结果
@st.cache_data(show_spinner=True)
def calc_solution(m, k, b, F0, omega):
    x0 = 2.0
    v0 = 0.0
    omega0 = np.sqrt(k / m)
    zeta = b / (2 * np.sqrt(k * m))
    def damped_forced_osc(t, y):
        x, v = y
        dxdt = v
        dvdt = (-b * v - k * x + F0 * np.cos(omega * t)) / m
        return [dxdt, dvdt]
    t_span = (0, 20)
    t_eval = np.linspace(*t_span, 1000)
    sol = solve_ivp(damped_forced_osc, t_span, [x0, v0], t_eval=t_eval)
    return sol, omega0, zeta

sol, omega0, zeta = calc_solution(m, k, b, F0, omega)

# 4. 显示参数和振动类型判别
col1, col2,col3 = st.columns(3)
col1.metric("固有频率 ω₀(rad/s)", f"{omega0:.3f}")
col2.metric("阻尼比 ζ", f"{zeta:.3f}")
col3.metric("外力频率 ω(rad/s)", f"{omega:.3f}")

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

# 5. 缓存动画生成函数
@st.cache_data(show_spinner=True)
def make_gif(sol):
    frames = []
    fig2, ax2 = plt.subplots(figsize=(5, 2))
    for i in range(0, len(sol.t), 5):
        ax2.clear()
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-1, 1)
        ax2.axis('off')
        x = sol.y[0][i]
        ax2.plot(x, 0, 'ro', markersize=20)
        ax2.plot(0, 0, 'ko', markersize=6)  # 中心点
        buf = BytesIO()
        fig2.savefig(buf, format='png')
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
        buf.close()
    plt.close(fig2)
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
        gif_path = tmpfile.name
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0
    )
    with open(gif_path, "rb") as f:
        gif_bytes = f.read()
    try:
        os.remove(gif_path)
    except Exception:
        pass
    return gif_bytes

gif_bytes = make_gif(sol)

col1, col2 = st.columns(2)
# 6. 显示动画
with col1:
    st.image(gif_bytes, caption="小球受迫振动动画", use_container_width=True)
    
# 7. 绘制位移-时间图
with col2:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(sol.t, sol.y[0], label="Displacement x(t)", color="blue")
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("x(t) (m)")
    ax.set_title("Displacement-Time Graph")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)