import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplcursors import cursor
import math


def add_nans(cals: list, dates: pd.Series) -> list:
    for cal in range(len(dates) - len(cals)):
        cals.append(np.nan)

    return cals

# Weight tracking
real_weight = [
               ] #### Add your weight to this list
est_cals = [
    
] #### Add your daily calories to this list
if len(real_weight) != len(est_cals):
    est_cals.append(np.nan)
cal_goal = 2000 #### Add daily calorie goal
high_cals = 2600 #### Add cheat day cutoff

# Get average value of calories by ignoring outliers
cals_no_outliers = [cal for cal in est_cals if cal <= high_cals]
avg_cal_nout = sum(cals_no_outliers) / len(cals_no_outliers)

# Create DataFrame
difference = np.diff(real_weight).tolist() + [0]
df = pd.DataFrame(data={'calories': est_cals, 'difference': difference, 'real_weight': real_weight})
avg_dif = df.difference.abs().mean()
avg_cal = df.calories.mean()

# Daily difference in weight
df = df.dropna(axis=0)
day = [f'day {i}' for i in range(len(df.index))]
df['day_1'] = day
max_day = df.loc[df['difference'] == df.difference.max(), 'day_1'].tolist()
min_day = df.loc[df['difference'] == df.difference.min(), 'day_1'].tolist()

# Cal days above set value
high_caldays = len(df.calories[df.calories >= high_cals])

# Plot difference and calories
ax_xticks = [i*10 for i in range(int(math.ceil(len(df.index)/10)))]

fig, ax1 = plt.subplots()
color='magenta'
ax1.set_xlabel('Days into Diet')
ax1.set_ylabel('Difference in Daily Weight (lbs)', color=color)
ax1.bar(df.day_1, df.difference, color=color, label='Difference in Weight (Today - Yesterday)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.axhline(y=0, linewidth=1, c='black', linestyle='-', label='_nolegend_')
ax1.axhline(
    y=avg_dif, linewidth=1, c='magenta', linestyle='--', label=f'Average Daily Difference ({round(avg_dif,2)} lbs)'
)
ax1.axhline(y=-avg_dif, linewidth=1, c='magenta', linestyle='--', label='_nolegend_')
ax1.set_ylim(bottom=-3, top=3)

ax2 = ax1.twinx()

color = 'tab:orange'
ax2.set_ylabel('Calories Per Day (kcal)', color=color)
ax2.scatter(df.day_1, df.calories, color=color, label='Caloric Intake')
ax2.plot(df.day_1, df.calories, color=color, label='_nolegend_')
ax2.tick_params(axis='y', labelcolor=color)
ax2.axhline(y=cal_goal, linewidth=1, c='black', linestyle='-', label=f'Target Calories: {cal_goal} cals')
ax2.axhline(y=avg_cal, linewidth=1, c='blue', linestyle='--',
            label=f'Average Calories (outliers removed): {int(avg_cal_nout)} cals')
ax2.set_ylim(bottom=cal_goal-800, top=cal_goal+800)

handles = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
plt.legend(lines, handles, loc='best', fontsize='x-small')
plt.xticks(ticks=ax_xticks, minor=False, rotation=45)
ax1.tick_params(axis='x', labelrotation=45)
cursor(hover=True)
#plt.savefig('diff.png', dpi=500)
plt.show()
