import matplotlib.pyplot as plt
import numpy as np

# Data from the provided results
data = {
    "First Runs": (11.149403892714401, 0.16758620689655174, 0.152758620689655),
    "Page Views * 2 in Title": (12.297557181325452, 0.3284137931034483, 0.38620689655172413),
    "Anchor Text Stem * 2": (15.345124888420106, 0.3259, 0.4133333333333334),
    "Title without Stemming and Bigram": (8.402171548207601, 0.40296666666666653, 0.5066666666666667),
    "Adding Threads": (7.330120650927226, 0.4104333333333333, 0.5399999999999999),
    "Removing Binary Title": (7.355520009994507, 0.4324, 0.5366666666666666),
    "Filter Low TF for Speed": (1.9250868082046508, 0.4325333333333333, 0.5399999999999999)
}

# Extracting data for plotting
groups = list(data.keys())
avg_retrieval_time = np.array([data[group][0] for group in groups])
avg_harmonic_mean = np.array([data[group][1] for group in groups])
avg_precision = np.array([data[group][2] for group in groups])

# Plot for average retrieval time
plt.figure(figsize=(10, 6))
plt.barh(groups, avg_retrieval_time, color='skyblue', edgecolor='black')
plt.title('Average Retrieval Time')
plt.xlabel('Time')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot for average harmonic mean
plt.figure(figsize=(10, 6))
plt.barh(groups, avg_harmonic_mean, color='salmon', edgecolor='black')
plt.title('Average Harmonic Mean')
plt.xlabel('Harmonic Mean')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot for average precision
plt.figure(figsize=(10, 6))
plt.barh(groups, avg_precision, color='lightgreen', edgecolor='black')
plt.title('Average Precision')
plt.xlabel('Precision')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
