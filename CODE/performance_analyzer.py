import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the results directory exists
output_dir = "results"
csv_path = os.path.join(output_dir, "performance_metrics.csv")

# Load the CSV file
df = pd.read_csv(csv_path)

# Convert timestamp to datetime if needed
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# -------------------------------
# Graph 1: Detection Count per Frame
plt.figure(figsize=(10,6))
sns.lineplot(x='frame_number', y='detection_count', data=df, marker="o")
plt.title("Detection Count per Frame")
plt.xlabel("Frame Number")
plt.ylabel("Detection Count")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "detection_count_per_frame.png"))
plt.show()

# -------------------------------
# Graph 2: Disaster Score per Frame
plt.figure(figsize=(10,6))
sns.lineplot(x='frame_number', y='disaster_score', data=df, marker="o", color="purple")
plt.title("Disaster Score per Frame")
plt.xlabel("Frame Number")
plt.ylabel("Disaster Score")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "disaster_score_per_frame.png"))
plt.show()

# -------------------------------
# Graph 3: Average Confidence Distribution
plt.figure(figsize=(10,6))
sns.histplot(df['avg_confidence'], kde=True, bins=20, color="green")
plt.title("Average Confidence Distribution")
plt.xlabel("Average Confidence")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "avg_confidence_distribution.png"))
plt.show()

# -------------------------------
# Graph 4: Processing Time per Frame
plt.figure(figsize=(10,6))
sns.lineplot(x='frame_number', y='processing_time', data=df, marker="o", color="red")
plt.title("Processing Time per Frame")
plt.xlabel("Frame Number")
plt.ylabel("Processing Time (s)")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "processing_time_per_frame.png"))
plt.show()

# -------------------------------
# Graph 5: Processing Time Breakdown (Bar Chart)
avg_processing_time = df['processing_time'].mean()
plt.figure(figsize=(6,6))
plt.bar(['Avg Processing Time'], [avg_processing_time], color="skyblue")
plt.title("Average Processing Time per Frame")
plt.ylabel("Time (s)")
plt.savefig(os.path.join(output_dir, "avg_processing_time_bar.png"))
plt.show()

# -------------------------------
# Graph 6: Processing Time Breakdown (Pie Chart)
# Note: Since the CSV logs only overall processing time, we assume a dummy breakdown.
# Replace these percentages with real component-level data if available.
components = ['YOLO Detection', 'DeepSort Tracking', 'Other']
percentages = [42, 28, 30]  # Dummy percentages, add up to 100 (or adjust as needed)
plt.figure(figsize=(6,6))
plt.pie(percentages, labels=components, autopct='%1.1f%%', startangle=140, colors=["#ff9999","#66b3ff","#99ff99"])
plt.title("Processing Time Breakdown")
plt.savefig(os.path.join(output_dir, "processing_time_breakdown_pie.png"))
plt.show()

# -------------------------------
# (Optional) Graph 7: Attention Weights per Frame
if 'attention_weight' in df.columns:
    plt.figure(figsize=(10,6))
    sns.lineplot(x='frame_number', y='attention_weight', data=df, marker="o", color="orange")
    plt.title("Attention Weight per Frame")
    plt.xlabel("Frame Number")
    plt.ylabel("Attention Weight")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "attention_weight_per_frame.png"))
    plt.show()
