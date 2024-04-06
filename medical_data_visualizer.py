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
    df_cat =  df.melt(id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = pd.DataFrame(df_cat.groupby(by=['cardio']).value_counts().reset_index().rename(columns={'count': 'total'}))
    
    # Draw the catplot with 'sns.catplot()'
    df_cat['value'] = df_cat['value'].replace({1:'1', 0:'0'})
    
    # Get the figure for the output
    fig = sns.catplot(data=df_cat, x='variable', y='total', hue='value', kind='bar', col='cardio')

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
#def draw_heat_map():
    # Clean the data
#    df_heat = None

    # Calculate the correlation matrix
#    corr = None

    # Generate a mask for the upper triangle
#    mask = None



    # Set up the matplotlib figure
#    fig, ax = None

    # Draw the heatmap with 'sns.heatmap()'



    # Do not modify the next two lines
#    fig.savefig('heatmap.png')
#    return fig
