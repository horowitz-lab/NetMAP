"""
My heatmap functions similarly to seaborn.heatmap but it makes a plot with
numeric axes.
"""
import matplotlib.pyplot as plt 


# Create a listwrap that wraps around the list 
# This is what I need for pcolormesh.
def listwrap(currentlist):
    # Ideally the dimensions of X and Y should be one greater than those of C; 
    # if the dimensions are the same, then the last row and column of C will be ignored.
    listwrap = [0] * (len(currentlist)+1)
    for i in range(len(currentlist)):
        try:
            listwrap[i+1] = ((currentlist[i]+currentlist[i+1])/2)
        except IndexError:
            pass
    topstep = listwrap[2]-listwrap[1]
    listwrap[0]=listwrap[1]-topstep
    botstep = listwrap[-2]-listwrap[-3]
    listwrap[-1]=listwrap[-2]+botstep
    return listwrap

# df is a pandas dataframe
def myheatmap(df, colorbarlabel=None, **kwargs):
    plt.pcolormesh( listwrap(df.columns),listwrap(df.index), df, **kwargs)
    plt.xlabel(df.columns.name)
    plt.ylabel(df.index.name)
    cbar = plt.colorbar(drawedges=False)
    if colorbarlabel:
        cbar.set_label(colorbarlabel)
    return plt.gca()
