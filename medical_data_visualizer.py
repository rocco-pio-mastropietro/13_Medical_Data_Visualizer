import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).replace({True: 0, False: 1})

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = (df['cholesterol'] > 1).replace({True: 1, False: 0})
df['gluc'] = (df['gluc'] > 1).replace({True: 1, False: 0})


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = df.melt(id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(by=['cardio']).value_counts().reset_index(name='total')

    # Draw the catplot with 'sns.catplot()'
    df_cat['value'] = df_cat['value'].replace({1:'1', 0:'0'})
    categorical_plot = sns.catplot(data=df_cat, x='variable', y='total', hue='value', kind='bar', col='cardio', order=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # Get the figure for the output
    fig = categorical_plot.figure

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.loc[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025))& (df['weight'] <= df['weight'].quantile(0.975))]
    
    # Calculate the correlation matrix
    corr = df_heat.corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(corr.to_numpy().reshape((14, 14)), k=0)

    # Set up the matplotlib figure
    fig, ax = fig, ax = plt.subplots()

    # Draw the heatmap with 'sns.heatmap()'
    heat_map = sns.heatmap(corr, mask=mask)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig