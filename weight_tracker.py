import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from mplcursors import cursor
from statistics import  mean


# Weight tracking
real_weight = [] #### Add your weight to this list
lw = len(real_weight)
dates = pd.date_range(start='01/01/2000', end='31/12/2000') #### Add date range
for weight in range(len(dates) - len(real_weight)):
    real_weight.append(np.nan)
goalw, startw = 100, 200 #### Add goal weight and starting weight
lin_weight = np.linspace(startw, goalw, len(dates))
lpw = round(7 * (lin_weight[0] - lin_weight[1]), 2)
df = pd.DataFrame(data={'date': dates, 'lin_weight': lin_weight, 'real_weight': real_weight})
df['goal'] = goalw
df['day'] = df.date
df['ordinal'] = pd.to_datetime(df['date']).apply(lambda date: date.toordinal())
df = df.set_index('date')
df['idx'] = df.index
startdt, enddt = df.index[0].date(), df.index[-1].date()

# Add linear regression of weight
x = df['ordinal'].head(lw).to_numpy()
y = df['real_weight'].head(lw).to_numpy()
m, b = np.polyfit(x, y, 1)
ct = m*(df['ordinal'].iloc[-1]) + b

# Weight plot
fig, axes = plt.subplots(nrows=1, ncols=1)

ax = df.plot(x='ordinal', y=['lin_weight', 'goal', 'real_weight'], color=['b', 'r', 'black'],
             xlabel='Date', ylabel='Weight (lbs)', kind='line', style=['--', '-', 'o'])
ax.axline(xy1=(0, b), slope=m, c='black', linestyle='-', label=f'$y = {m:.1f}x {b:+.1f}$')
ax.legend(['Ideal weight-loss tajectory', 'Goal weight', 'Measured weight', 'Current real weight-loss trajectory'])
ax.set_ylim(goalw - 3, startw + 3)
ax.set_xlim(df['ordinal'].min() - 1, df['ordinal'].max() + 1)

yticks = np.arange(goalw - 3, startw, 1)
ax.set_yticks(ticks=yticks, minor=True)
new_labels = [date.fromordinal(int(item)) for item in ax.get_xticks()]
ax.set_xticks(df.ordinal, minor=True)
ax.set_xticklabels(new_labels, rotation=45)

ax.set_title(f'Weight Loss From {startdt} to {enddt}\nMeasured After Urination in the Morning')
if m < 0 and m*7 < lpw:
    ax.text(x=0.02, y=0.3, s=f'Weekly weight loss needed to hit goal:\n{lpw} lbs/week'
                              f'\n\nYou are currently LOSING weight at a pace of:\n{abs(round(m*7,2))} lbs/week'
                              f'\n\nWeight at goal date according to current trajectory:\n{round(ct,2)} lbs'
                              f'\n\nYou will miss your goal by {abs(round(goalw - ct, 2))} lbs',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            bbox=dict(facecolor='red', edgecolor='black', pad=10.0, alpha=0.1))
if m < 0 and m*7 > lpw:
    ax.text(x=0.02, y=0.3, s=f'Weekly weight loss needed to hit goal:\n{lpw} lbs/week'
                              f'\n\nYou are currently LOSING weight at a pace of:\n{abs(round(m*7,2))} lbs/week'
                              f'\n\nWeight at goal date according to current trajectory:\n{round(ct,2)} lbs'
                              f'\n\nYou will beat your goal by {abs(round(ct - goalw, 2))} lbs',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            bbox=dict(facecolor='red', edgecolor='black', pad=10.0, alpha=0.1))
if m < 0 and m*7 == lpw:
    ax.text(x=0.02, y=0.3, s=f'Weekly weight loss needed to hit goal:\n{lpw} lbs/week'
                              f'\n\nYou are currently LOSING weight at a pace of:\n{abs(round(m*7,2))} lbs/week'
                              f'\n\nWeight at goal date according to current trajectory:\n{round(ct,2)} lbs'
                              f'\n\nYou will nail your goal!',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            bbox=dict(facecolor='red', edgecolor='black', pad=10.0, alpha=0.1))
elif m > 0:
    ax.text(x=0.02, y=0.3, s=f'Weekly weight loss needed to hit goal:\n{lpw} lbs/week'
                              f'\n\nYou are currently GAINING weight at a pace of:\n{abs(round(m*7, 2))} lbs/week'
                              f'\n\nWeight at goal date according to current trajectory:\n{round(ct, 2)} lbs'
                              f'\n\nYou will miss your goal by {abs(round(goalw - ct, 2))} lbs',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            bbox=dict(facecolor='red', edgecolor='black', pad=10.0, alpha=0.1))
ax.grid(visible=True, which='major', axis='both', c='black', linestyle='--', linewidth=2, alpha=0.1)

# Activity tracking
steps = [] #### Add daily steps for approximate NEAT
avg_steps = mean(steps)
step_error = mean([step * 1.15 for step in steps]) - avg_steps #### Assumes 15% error on step count

for step in range(len(dates) - len(steps)):
    steps.append(np.nan)
df['steps'] = steps
df['step_err'] = df['steps'] * 0.15

# Step plot
ax1 = df.plot.bar(use_index=True, y='steps', color='b', alpha=0.5, yerr='step_err',
                  error_kw={'ecolor': 'black', 'elinewidth': 0.3, 'capsize': 1},
                  xlabel='Date', ylabel=r'Daily Steps (± 15%)', legend=True)
ax1.axhline(y=avg_steps, linewidth=1, color='r', linestyle='--', label='average daily steps')
ax1.legend(['Average daily steps', 'Daily steps'])
ax1.set_title(f'Daily Steps From {startdt} to {enddt}\nMeasured from Phone between 00:00 to 23:59')
ax1.text(x=0.82, y=0.84, s=f'Average daily steps:\n'
                          f'{int(avg_steps)} ± {int(step_error)} steps/day',
         horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes,
         bbox=dict(facecolor='none', edgecolor='black', pad=10.0, alpha=0.15))
ax1.set_ylim(0, int(max(steps) * 1.25))

ticklabels = [''] * len(df)
skip = len(df) // 12
ticklabels[::skip] = df['day'].iloc[::skip].dt.strftime('%m-%d')
ax1.xaxis.set_major_formatter(mticker.FixedFormatter(ticklabels))
fig.autofmt_xdate()


# Fixes the tracker
# https://matplotlib.org/users/recipes.html
def fmt(x, pos=0, max_i=len(ticklabels) - 1):
    i = int(x)
    i = 0 if i < 0 else max_i if i > max_i else i
    return dates[i]


ax1.fmt_xdata = fmt

plt.xticks(rotation=45)
cursor(hover=True)
plt.show()
