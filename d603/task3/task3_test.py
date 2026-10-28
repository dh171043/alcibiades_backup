import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

pd.set_option('display.max_columns', None) # view all columns when printing
spacer = "*"*20
def header_txt(message):
    output = f"\n{spacer} {message} {spacer}\n"
    print(output)
churn = pd.read_csv("C:/CraneDev/python_projects/wgu/d603/task3/churn_clean.csv")
print(churn.head())


header_txt("Data Prepration")
churn['Day'] = pd.to_datetime(churn["Day"])
churn = churn.set_index("Day")

header_txt("Visualization")
fig, ax = plt.subplots(figsize = (16, 7))
ax.plot(churn.index, churn['Revenue'], label = "Daily Revenue")
ax.set_title('Daily Revenue Over Time', fontsize=18, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Revenue ($)', fontsize=12)

# Format the x-axis to display dates nicely
# This locator and formatter will make the date labels clean and prevent overlapping.
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right') # Rotate date labels

# Add a grid for easier reading of values
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend
ax.legend()

# Ensure everything fits without overlapping
plt.tight_layout()


plt.savefig("C:/CraneDev/python_projects/wgu/d603/task3/time_step_realization")

