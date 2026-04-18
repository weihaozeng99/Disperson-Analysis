


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Upload raw data file in csv

df = pd.read_csv("./data.csv", header=None)

stat = df.agg(['mean', lambda x: x.std(ddof=0)]).rename(index={'<lambda>': 'std'})

df_normalized = (df - df.mean()) / df.std(ddof=0)
    
    # Check normalized statistics
norm_stat = df_normalized.agg(['mean', lambda x: x.std(ddof=0)]).rename(index={'<lambda>': 'std'})

    # Compute mean of every 2 rows with improved accuracy
df_block_means = []
for i in range(0,9):
    df_block_mean = df.groupby(df.index // (2**i)).agg(lambda x: np.mean(x.values, dtype=np.float64))
    df_block_means.append(df_block_mean.std(ddof=0))

df_block_means_log2 = [np.log2(block) for block in df_block_means]
print(df_block_means_log2[0][1])

x_plot = [0,1,2,3,4,5,6,7,8]

df_log2_df = pd.DataFrame(df_block_means_log2, index=x_plot)

x = np.array(x_plot, dtype=float)

fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

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
plt.show()

ax = df_log2_df.plot(marker='o', figsize=(8, 5))
ax.set_xlabel("i")
ax.set_ylabel("log2(std)")
ax.set_title("log2 block std by column")
plt.tight_layout()
plt.show()