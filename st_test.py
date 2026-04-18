import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def polyresid(data, order):
    """模拟 MATLAB polyresid: 移除多项式趋势"""
    if order <= 0:
        return data
    x = np.arange(len(data))
    coeffs = np.polyfit(x, data, order)
    poly_curve = np.polyval(coeffs, x)
    return data - poly_curve

def normalize_series(data):
    """将序列标准化为均值0，标准差1"""
    return (data - np.mean(data)) / np.std(data, ddof=1)

st.title('Dispersion Analysis (Advanced Preprocessing)')

# --- 侧边栏：实验参数与过滤设置 ---
st.sidebar.header("1. Experimental Constraints")
p_isi = st.sidebar.number_input("ISI (ms)", value=200)
p_stim = st.sidebar.number_input("Stimulus (ms)", value=972)
p_timeout = st.sidebar.number_input("Timeout Threshold (ms)", value=976)

st.sidebar.header("2. MATLAB-style Censoring")
# 第一步：绝对值截断
lo_val = st.sidebar.number_input("Min Value (Absolute)", value=0.0)
hi_val = st.sidebar.number_input("Max Value (Absolute)", value=2000.0)

# 第二步：标准差去噪
sd_threshold = st.sidebar.slider("SD Threshold (n * sigma)", 1.0, 5.0, 3.0)

# 第三步：长度调整
target_len = st.sidebar.selectbox("Target Length (Power of 2)", [512, 1024, 2048, 4096], index=1)

# 第四步：去趋势阶数
detrend_order = st.sidebar.slider("Detrend Polynomial Order", 0, 4, 1)

# --- 数据上传 ---
raw_data = st.file_uploader("Upload Raw Data File (CSV)", type=["csv"])

if raw_data is not None:
    try:
        # 加载数据
        df_raw = pd.read_csv(raw_data, header=None)
        st.write(f"Original Data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

        processed_cols = {}

        for col in df_raw.columns:
            series = df_raw[col].dropna().values
            
            # --- 逻辑过滤 1: 实验超时过滤 ---
            series = series[series <= p_timeout]
            
            # --- 逻辑过滤 2: 绝对值截断 (Absolute Censoring) ---
            series = series[(series >= lo_val) & (series <= hi_val)]
            
            # --- 逻辑过滤 3: 标准差去噪 (SD Censoring) ---
            mu, sigma = np.mean(series), np.std(series, ddof=1)
            series = series[np.abs(series - mu) <= sd_threshold * sigma]
            
            # --- 逻辑过滤 4: 去趋势与标准化 ---
            series = polyresid(series, detrend_order)
            series = normalize_series(series)
            
            # --- 逻辑过滤 5: 长度调整 (Truncation/Zero-padding) ---
            current_len = len(series)
            if current_len > target_len:
                # 按照 MATLAB 逻辑：如果太长，截掉开头的数据
                series = series[current_len - target_len:]
            elif current_len < target_len:
                # 如果太短，末尾补零
                padding = np.zeros(target_len - current_len)
                series = np.concatenate([series, padding])
            
            processed_cols[f"Col_{col}"] = series

        df_final = pd.DataFrame(processed_cols)
        st.success(f"Preprocessing Complete. Final shape: {df_final.shape}")
        st.dataframe(df_final.head())

        # --- 离散度分析 (Dispersion Analysis) ---
        st.subheader("Dispersion Analysis & Fractal Dimension")
        
        x_plot = np.arange(int(np.log2(target_len)) - 1) # 动态计算缩放等级
        results = []

        for col in df_final.columns:
            data_col = df_final[col].values
            stds = []
            for i in x_plot:
                block_size = 2**i
                # 分组求均值后计算标准差
                reshaped = data_col[:(len(data_col)//block_size)*block_size].reshape(-1, block_size)
                block_means = np.mean(reshaped, axis=1)
                stds.append(np.std(block_means, ddof=0))
            
            y_log = np.log2(np.array(stds) + 1e-10) # 防止log(0)
            
            # 线性拟合
            slope, intercept = np.polyfit(x_plot, y_log, 1)
            results.append({"Column": col, "Slope": slope, "FD": 1 - slope})
            
            # 绘图
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(x_plot, y_log, 'o-', label=f"FD={1-slope:.3f}")
            ax.set_title(f"Analysis for {col}")
            ax.set_xlabel("Log2(Scale)")
            ax.set_ylabel("Log2(Std)")
            ax.legend()
            st.pyplot(fig)

        st.table(pd.DataFrame(results))

    except Exception as e:
        st.error(f"Error: {e}")