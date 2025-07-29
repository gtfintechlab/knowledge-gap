import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/poc_revenue_mcap_data_v2.csv')

# Keep data only starting 1980
df = df.loc[(df['year'] >= 1980) & (df['year'] <= 2022)]

print(df.shape)



#################### More interesting plot #####################

df = df.sort_values(by="year", ascending=True, ignore_index=False)

df['First'] = df.duplicated('GVKEY', keep='first')
df['Last'] = df.duplicated('GVKEY', keep='last')

# to handle companies which only appear once and mark them as last only 
df.loc[(df['First'] == False) & (df['Last'] == False), 'First'] = True

# df.to_csv("temp.csv")

yearly_counts = df.groupby('year').agg(First=('First', lambda x: (~x).sum()),
                                       Last=('Last', lambda x: (~x).sum()))

yearly_counts['Total'] = df.groupby('year')['GVKEY'].nunique()
yearly_counts['Remaining'] = yearly_counts['Total'] - (yearly_counts['First'] + yearly_counts['Last'])


yearly_counts.drop(columns=['Total'], inplace=True)

print(yearly_counts)

'''
yearly_counts.plot(kind='bar', stacked=True)
plt.title('Number of Companies Over Years')
plt.xlabel('Years')
plt.ylabel('Number of Companies')
# plt.show()

ticks_to_show = yearly_counts.index[::5]  # Change 2 to another number to adjust the skipping

# Set the x-ticks
plt.xticks(ticks=range(len(yearly_counts.index)), labels=yearly_counts.index, rotation=45)
# Now hide every other tick label to reduce clutter
for i, label in enumerate(plt.gca().get_xticklabels()):
    if i % 5 != 0:  # Adjust this condition to change which ticks are shown
        label.set_visible(False)

plt.rcParams['font.size'] = 25

# Show the plot
plt.tight_layout()
plt.show()
'''



# Re-plot the bar graph with updated figure size
yearly_counts.plot(kind='bar', stacked=True)

# Update the title and axis labels
plt.title('Number of Companies Over the Years', fontsize=20)
plt.xlabel('Years', fontsize=16)
plt.ylabel('Number of Companies', fontsize=16)

# Update the legend to use clearer terms
plt.legend(['First Appearance', 'Final Appearance', 'Ongoing'], fontsize=14)

# Adjust the x-ticks to display every 5th year and rotate for better visibility
plt.xticks(ticks=range(len(yearly_counts.index)), labels=yearly_counts.index, rotation=30)
for i, label in enumerate(plt.gca().get_xticklabels()):
    if i % 5 != 0:  
        label.set_visible(False)

# Set larger font sizes for clarity
plt.rcParams['font.size'] = 16

# Ensure the layout is tight and better fit
plt.tight_layout()

# Increase the size of the figure to make it wider
plt.figure(figsize=(12, 6))  # Adjust the width and height as needed

# Show the updated plot
plt.show()