import pandas as pd
import tester as tst

from pylab import savefig

## VARIABLES
# startt = spectrogram label location start times
# endt = spectrogram label location end times
# desc = whale song label descriptions
# nentries = number of labels in a .txt file


## Loads up the files to parse
def loader():
    #### TODO: add the automated file loading process here
    return pd.read_table('20151023T122324 copy.txt')

    #### TODO: load the associated spectrogram file too! 
    #### CHECK LINE 35 OF tester.py TO SEE .WAV LOCATION



## Creates the file names for the spectrogram slices by transforming the 
## float variables to strings.
def nconvention(name):
    
    # convert float to string  
    name[0] = '%.1f' % name[0]
    name[1] = '%.1f' % name[1]

    # remove blank spaces and + signs, replace with _
    name[2] = name[2].replace(" ", "_")
    name[2] = name[2].replace("+", "_")

    return name



## Finds the start time, end time, and label variables from the associated .txt file,
## then creates spectrogram slices of each label.
def splicer(df):
    name = [0,1,2]

    nentries = df.shape[0]-1    
    for i in range(1):
        startt = df.iloc[i]['Begin Time (s)']
        endt = df.iloc[i]['End Time (s)']
        desc = df.iloc[i]['Description']
        
        name[0] = startt
        name[1] = endt
        name[2] = desc
        nconvention(name)

        #### TODO: make sure the file name is reflected in the naming convention
        #finaln = "20151023T122324_" + str(i+1) + "_" + name[2] + "_" + name[0] + "_" + name[1]
        finaln = name[2] + "_" + name[0] + "_" + name[1]
        tst.sgraph(startt,endt,finaln)

def main():
    df = loader()
    splicer(df)
    
if __name__ == "__main__":
    main()
