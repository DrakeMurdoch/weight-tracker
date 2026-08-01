import pandas as pd
import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt
from mplcursors import cursor
from diff_refactored import avg_cal_nout, cals_no_outliers

def add_nans(input_list: list, target_size: int) -> list:
    for cal in range(target_size + 1 - len(input_list)):
        input_list.append(np.nan)
    return input_list

# Weight tracking
real_weight = [

] #### Add your weight to this list
lw = len(real_weight)
goalw, startw = 200, real_weight[0] #### Add goal weight here
noof_days = list(range(lw))

# Linear regression of weight over time to predict weight loss trend
x0 = noof_days
y0 = real_weight
m0, b0 = np.polyfit(x0, y0, 1)

total_days = int((goalw - b0)/m0)
start_date = date(2026, 01, 01) #### Add start date
end_date = start_date + timedelta(days=total_days)
lin_weight0 = [m0*i+b0 for i in range(total_days+1)]

# Remove outliers and recalculate linear regression
real_vs_lin = [float(i-j) for i,j in zip(real_weight, lin_weight0)]
max_diff = 0.8 #### This value was accurate for me, but can be adjusted to sync up trendline with real values
indices = []
diff_real_weight = real_weight.copy()
diff_noof_days = noof_days.copy()
for r, d in enumerate(real_vs_lin):
    if d >= max_diff:
        indices.append(r)
for index in sorted(indices, reverse=True):
    del diff_real_weight[index]
    del diff_noof_days[index]

x = diff_noof_days
y = diff_real_weight
m, b = np.polyfit(x, y, 1)
lin_weight = [m*i+b for i in range(total_days+1)]

# Calculate daily average calories burnt
daily_excess_cals = -m*3600
cals_burnt = int(daily_excess_cals + avg_cal_nout)
cals_std = int(np.std(cals_no_outliers))

# List with only dropped values
dropped_weights = []
for num in range(lw):
    if num in indices:
        dropped_weights.append(real_weight[num])
    else:
        dropped_weights.append(np.nan)

# Linear regression of last just week's weights
x1 = noof_days[-7:]
y1 = real_weight[-7:]
m1, b1 = np.polyfit(x1, y1, 1)
lin_weight1 = [m1*i+b1 for i in range(total_days+1)]

# Turn it all into a pandas DataFrame
dates = pd.date_range(start=start_date, end=end_date)
dropped_weights_nan = add_nans(dropped_weights.copy(), total_days)
real_weight_nan = add_nans(real_weight.copy(), total_days)
df = pd.DataFrame(data={
        'date': dates,
        'lin_weight': lin_weight,
        'real_weight': real_weight_nan,
        'last_week': lin_weight1,
        'dropped': dropped_weights_nan
})
df['goal'] = goalw
df['day'] = df.date
df['ordinal'] = pd.to_datetime(df['date']).apply(lambda date: date.toordinal())
df = df.set_index('date')
df['idx'] = df.index
projected_weight = round(df.lin_weight.iloc[len(noof_days)],1)
goal_date = df.day.iloc[-1]

today = df.day.iloc[lw].date()
days_left = len(df.goal) - lw

# Find milestone date
milestone = 210 #### Add whatever milestone weight you want
true_milestone = min(lin_weight, key=lambda x:abs(x-milestone))
milestone_index = lin_weight.index(true_milestone)
milestone_ord = df.ordinal.iloc[milestone_index]
milestone_date = df.day.iloc[milestone_index]

# Plot everything relevant
fig, ax = plt.subplots(nrows=1, ncols=1)

ax.plot(df['ordinal'], df['lin_weight'], color='b', linestyle='-')
ax.plot(df['ordinal'], df['goal'], color='r', linestyle='-')
ax.scatter(df['ordinal'], df['real_weight'], color='black', s=10, marker='X', alpha=0.9)
ax.scatter(df['ordinal'], df['dropped'], color='r', s=10, marker='X', alpha=1)
ax.plot(df['ordinal'], df['last_week'], color='black', linestyle='--')
ax.vlines(x=milestone_ord, ymin=goalw-3, ymax=startw, color='purple', linestyle='dotted')
ax.legend([f'Overall Weight-loss Trajectory ({round(m*7,2)} lbs/week)', f'Goal Weight ({goalw} lbs)',
        'Measured Weight', 'Measurements Dropped from Regression',
        f'7-Day Weight-loss Trajectory ({round(m1*7,2)} lbs/week)',
        f'Milestone Weight Date ({milestone} lbs on {milestone_date.date()})'], fontsize=7)
ax.set_ylim(goalw - 3, startw + 3)
ax.set_xlim(df['ordinal'].min() - 1, df['ordinal'].max() + 1)

yticks = np.arange(goalw - 3, startw, 1)
ax.set_yticks(ticks=yticks, minor=True)
new_labels = [date.fromordinal(int(item)) for item in ax.get_xticks()]
ax.set_xticks(df.ordinal, minor=True)
ax.set_xticklabels(new_labels, rotation=45)

ax.set_title(f'Weight Loss From {start_date} to {end_date} ({len(df.goal)} days)\n'
             f'Measured After Urination in the Morning ({today})')
ax.text(
    x=0.03, y=0.19,
    s=f'Total weight loss: {round(startw - min(real_weight[:lw]),1)} lbs lost\n'
      f'Current Measured Weight: {real_weight[lw-1]} lbs\n'
      f'Trendline Projected Weight: {projected_weight} lbs\n'
      f'Projected Goal Date: {goal_date.date()} ({days_left} days until goal)\n'
      f'Average Daily Total Caloric Burn: {cals_burnt} ± {cals_std}',
    horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8,
    bbox=dict(facecolor='red', edgecolor='black', pad=4, alpha=0.1)
)
ax.grid(visible=True, which='major', axis='both', c='black', linestyle='--', linewidth=2, alpha=0.1)

plt.xticks(rotation=30)
cursor(hover=True)
#plt.savefig('tracker.png', dpi=500)
plt.show()
