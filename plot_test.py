import pandas as pd
import matplotlib.pyplot as plt

def make_fig(df, plotpath='tmp.png'):
    fig, axs = plt.subplots(2,1)
    axs = axs.flatten()
    axs[0].plot(df.timestamp, df.x, 'r')
    axs[0].plot(df.timestamp, df.y, 'k')
    axs[1].scatter(df.x, df.y, c=df.timestamp)
    plt.savefig(plotpath)

df = pd.read_csv('serout.csv', header=None)
df.columns = ['x', 'y', 'timestamp', 'empty']

make_fig(df, 'raw.png')
make_fig(df.rolling(15).median(), 'median15.png')
make_fig(df.rolling(15).mean(), 'mean15.png')
    
