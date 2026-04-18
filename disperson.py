import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title('Dispersion Analysis')

x_plot = [0,1,2,3,4,5,6,7,8]

# Upload raw data file in csv
raw_data = st.file_uploader("Upload Raw Data File in CSV format")
if raw_data is not None:
    try:
        df = pd.read_csv(raw_data, header=None)
        for col in df.columns:
            if df[col].count() != 1024:
                st.warning(f'Column {col+1} does not have 1024 rows')
        
        st.write('Raw Data Uploaded Successfully, total columns: ', len(df.columns))

    
    except Exception as e:
        st.error(e)
        st.stop()

    # Process data and plot
    stat = df.agg(['mean', lambda x: x.std(ddof=0)]).rename(index={'<lambda>': 'std'})
    st.write('Data Statistics')
    st.dataframe(stat)
    

    # Normalize the data using the computed mean and std
    df_normalized = (df - df.mean()) / df.std(ddof=0)

    
    # # Check normalized statistics
    # norm_stat = df_normalized.agg(['mean', lambda x: x.std(ddof=0)]).rename(index={'<lambda>': 'std'})
    # st.write('Normalized Data Statistics')
    # st.dataframe(norm_stat)

    df_block_means = []
    # Compute mean of  rows with different group sizes
    for i in range(0,9):
        df_block_mean = df.groupby(df.index // (2**i)).agg(lambda x: np.mean(x.values, dtype=np.float64))
        df_block_means.append(df_block_mean.std(ddof=0))
    
    df_block_means_log2 = [np.log2(block) for block in df_block_means]
    
    df_log2_df = pd.DataFrame(df_block_means_log2, index=x_plot)

    x = np.array(x_plot, dtype=float)
    fig_all, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

    for col, ax in zip(df_log2_df.columns, axes):
        y = df_log2_df[col].values.astype(float)

        # fit a line
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x + intercept

        # R^2
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        ax.plot(x, y, 'o', label='data')
        ax.plot(x, y_fit, '-', label=f'fit: y={slope:.3f}x+{intercept:.3f}, $R^2$={r2:.3f}')
        ax.set_ylabel(f"Column {col}")
        ax.legend(loc='best')

    axes[-1].set_xlabel("i")
    plt.tight_layout()
    st.pyplot(fig_all)

    fig_one, ax_one = plt.subplots(figsize=(8, 5))
    df_log2_df.plot(marker='o', linestyle='None', ax=ax_one)
    
    # Add linear regression lines for each column
    for col in df_log2_df.columns:
        y = df_log2_df[col].values.astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x + intercept
        ax_one.plot(x, y_fit, '-', label=f'Column {col}: slope={slope:.3f}')
    
    ax_one.set_xlabel("i")
    ax_one.set_ylabel("log2(std)")
    ax_one.set_title("Date with linear fits")
    ax_one.legend()
    plt.tight_layout()
    st.pyplot(fig_one)


    
    
    

    
