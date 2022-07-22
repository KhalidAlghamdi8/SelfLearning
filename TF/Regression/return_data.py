import pandas as pd
import numpy as np

def read_goog_msft():

    googFilename = "TF\datasets\GOOG.csv"
    msftFilename = "TF\datasets\MSFT.csv"

    goog = pd.read_csv(googFilename, sep=',', usecols=[0,5], header=0, names=['Data', 'Goog'])
    msft = pd.read_csv(msftFilename, sep=',', usecols=[0,5], header=0, names=['Data', 'MSFT'])

    goog['MSFT'] = msft['MSFT']

    goog['Data'] = pd.to_datetime(goog['Data'], format='%Y-%m-%d')

    goog = goog.sort_values(['Data'], ascending=[True])

    returns = goog[[Key for Key in dict(goog.dtypes) if dict(goog.dtypes) [Key] in ['float64', 'int64']]].pct_change()

    xData = np.array(returns['Goog'])[1:]
    yData = np.array(returns['MSFT'])[1:]
    return(xData, yData)