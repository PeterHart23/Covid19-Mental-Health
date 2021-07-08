import os
import sys
from pathlib import Path
import numpy as np
import pickle
import math
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.patches as mpatches
import plotly.figure_factory as ff

from OhioSort import month, year

month = '10'
year = '2020'
url = f'Datasets/States/Ohio/Ohio{month}_{year}.csv'
data_path = Path(url).resolve()
data = pd.read_csv(data_path)
statedata = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/minoritymajority.csv')
ohiodata = statedata[statedata['STNAME'] == 'Ohio']
fips = ohiodata['FIPS'].tolist()
countynames = ohiodata['CTYNAME'].tolist()
cs = [
        'rgb(204,0,0)',
        'rgb(255,51,51)',
        'rgb(255,102,102)',
        'rgb(255,153,153)',
        'rgb(255, 240, 240)',
    ]
cs = cs[::-1]

def main():
    countysentiment()



def countyemotion():
    global data, month, year, url, data_path,fips,countynames,cs
    anger = fear = joy = sadness = 0
    emlist = ['anger','fear','joy','sadness']
    # months20 = ['03','04','05','06','07','08','09']
    year = '2019'
    time = "after"
    Ohioemscores = pd.DataFrame(columns=['TweetID','Date','TweetMessage'])

    if time == "before":
        months19 = ['07','08','09','10','12']
        months20 = ['01','02']
        year = '2019'
        for i in months19:
            Ohiourl = f'Datasets/States/Ohio/Ohio{i}_{year}.csv'
            Ohio_path = Path(Ohiourl).resolve()
            Ohiodata = pd.read_csv(Ohio_path)
            Ohioemscores = Ohioemscores.append(Ohiodata,sort=False)

    elif time == "after":
        months20 = ['03','04','05','06','07','08','09']

    year = '2020'
    for i in months20:
        Ohiourl = f'Datasets/States/Ohio/Ohio{i}_{year}.csv'
        Ohio_path = Path(Ohiourl).resolve()
        Ohiodata = pd.read_csv(Ohio_path)
        Ohioemscores = Ohioemscores.append(Ohiodata,sort=False)

    for i in range(88): #88 counties * 4 emotions
        for x in range(4):
            Ohioemscores = Ohioemscores.append({'County':countynames[i],'emotion':emlist[x]},ignore_index=True)


    # group = data.groupby(['Year','Month','emotion']).count()['Date']
    # group = data.groupby(['County','Year','Month','emotion']).count()['Date']
    group = Ohioemscores.groupby(['County','emotion']).count()['Date']

    anggroup = group[0::4] #get all the anger values for a certain month for each county
    feargroup = group[1::4]
    joygroup = group[2::4]
    sadgroup = group[3::4]
    print(len(Ohioemscores))
    print(sum(joygroup))
    emotionswitch = {
        0: "anger",
        1: "fear",
        2: "joy",
        3: "sadness",
    }
    emotionlist = []

    for i in range(len(anggroup)):
        max = 0
        count = "anger"
        max = anggroup[i]
        if feargroup[i] > max:
            max = feargroup[i]
            count = "fear"
        if joygroup[i] > max:
            max = joygroup[i]
            count = "fear";
        if sadgroup[i] > max:
            max = sadgroup[i]
            count = "sadness";
        emotionlist.append(count)

    # colorscale = ["red","orange","green","blue"]
    endpts = [1,500,1000,2000]

    fig = ff.create_choropleth(
        fips=fips, values=feargroup, scope=['Ohio'], show_state_data=True,
        colorscale=cs,  round_legend_values=True,
        plot_bgcolor='rgb(229,229,229)',
        paper_bgcolor='rgb(229,229,229)',
        title = 'Fear before Covid-19',
        legend_title='Amount of Fear Tweets',
        county_outline={'color': 'rgb(255,255,255)', 'width': 0.5},
        exponent_format=True,
        binning_endpoints=endpts

    )
    fig.layout.template = None
    # fig.show()
    # fig.write_image(f"Datasets/States/Ohio/EmotionImages/beforeEmotion.png")





    # anggroup.plot()
    # feargroup.plot()
    # joygroup.plot()
    # sadgroup.plot()
    #
    # anger_patch = mpatches.Patch(color='red', label='Anger')
    # fear_patch = mpatches.Patch(color='orange', label='Fear')
    # joy_patch = mpatches.Patch(color='green', label='Joy')
    # sad_patch = mpatches.Patch(color='blue', label='Sadness')
    #
    # plt.legend(handles=[anger_patch,fear_patch,joy_patch,sad_patch])
    # plt.ylabel('Number of Tweets')
    # plt.xlabel('Year, Month')
    # plt.show()

        #plotting averages by month
        # monthvalues = dataframe.groupby(['Year','Month']).mean()['Raw Score']
        # monthvalues.Date.plot()

        # print(monthvalues)
        #

        #plotting averages by day
        # dayaverages = dataframe.groupby(['Day']).mean()['Raw Score']
        # dayaverages.plot()
        # plt.show()


#sentiment by county
def countysentiment():
    global data, month, year, url, data_path, fips, countynames,cs
    time = "before"
    Ohiosentscores = pd.DataFrame(columns=['TweetID','Date','TweetMessage'])

    if time == "before":
        months19 = ['07','08','09','10','12']
        months20 = ['01','02']
        year = '2019'
        for i in months19:
            Ohiourl = f'Datasets/States/Ohio/Ohio{i}_{year}.csv'
            Ohio_path = Path(Ohiourl).resolve()
            Ohiodata = pd.read_csv(Ohio_path)
            Ohiosentscores = Ohiosentscores.append(Ohiodata,sort=False)

    elif time == "after":
        months20 = ['03','04','05','06','07','08','09']

    year = '2020'
    for i in months20:
        Ohiourl = f'Datasets/States/Ohio/Ohio{i}_{year}.csv'
        Ohio_path = Path(Ohiourl).resolve()
        Ohiodata = pd.read_csv(Ohio_path)
        Ohiosentscores = Ohiosentscores.append(Ohiodata,sort=False)

    #create filler data so every county has at least 1 datapoint. needed to equal fips number
    for i in range(88): #88 counties * 4 emotions
            Ohiosentscores = Ohiosentscores.append({'County':countynames[i],'Sentiment':-1},ignore_index=True)

    print()
    # sentvalues = Ohiosentscores.groupby(['County']).mean()['Raw Score']

    df_filtered = Ohiosentscores.query('Sentiment == 1')
    sentvalues = df_filtered.groupby(['County']).count()['Sentiment']
    print(sentvalues)
    # sentvalues = Ohiosentscores.groupby(['County']).count()['Sentiment']

    print(sum(sentvalues))
    # print(sentvalues)
    # sentvalues = (sentvalues + 2) * 10
    # for i in range(len(sentvalues)):
    #     if math.isnan(sentvalues[i]):
    #         sentvalues[i] = 25
    # endpts = list(np.mgrid[min(sentvalues):max(sentvalues):4j])
    # endpts = [1,3000,6000,9000]
    # fig = ff.create_choropleth(
    #     fips=fips, values=sentvalues, scope=['Ohio'], show_state_data=True,
    #     colorscale=cs,  round_legend_values=True,
    #     plot_bgcolor='rgb(229,229,229)',
    #     paper_bgcolor='rgb(229,229,229)',
    #     title = 'Sentiment after Covid-19',
    #     legend_title='Amount of Negative Tweets',
    #     county_outline={'color': 'rgb(255,255,255)', 'width': 0.5},
    #     exponent_format=True,
    #     # binning_endpoints=[-0.15,-0.05,0,0.05,0.1,0.15],
    #     binning_endpoints=endpts
    # )
    # fig.layout.template = None
    # fig.show()
    # fig.write_image(f"Datasets/States/Ohio/SentImages/afterCount.png")

# #plot Countries with line graph
def countrysentiment():
    months19 = ['07','08','09','10','12']
    months20 = ['01','02','03','04','05','06','07','08','09']

    year = '2019'
    USsentscores = []
    for i in months19:
        USurl = f'Datasets/States/US/US{year}_{i}.csv'
        US_path = Path(USurl).resolve()
        USdata = pd.read_csv(US_path)
        USvals = USdata['Raw Score'].mean()
        USsentscores.append(USvals)
    year = '2020'
    for i in months20:
        USurl = f'Datasets/States/US/US{year}_{i}.csv'
        US_path = Path(USurl).resolve()
        USdata = pd.read_csv(US_path)
        USvals = USdata['Raw Score'].mean()
        USsentscores.append(USvals)

    Canurl = f'Datasets/Canada/AllCanada.csv'
    Can_path = Path(Canurl).resolve()
    Candata = pd.read_csv(Can_path)
    # Cangroup = Candata.groupby(['Year','Month']).mean()['Raw Score']
    # ca_filtered = Candata.query('Sentiment == 1')
    Cangroup = Candata.groupby(['Year','Month']).count()['Sentiment']

    Cansentscores = []
    for i in Cangroup:
        Cansentscores.append(i)
    Cansentscores.pop()
    print(sum(Cansentscores[0:7]))
    print(sum(Cansentscores[8:15]))
    # print(sum(Cansentscores))

    Mexurl = f'Datasets/AllMexico.csv'
    Mex_path = Path(Mexurl).resolve()
    Mexdata = pd.read_csv(Mex_path)
    Mexgroup = Mexdata.groupby(['Year','Month']).mean()['Raw Score']

    # mx_filtered = Mexdata.query('Sentiment == 1')
    # Mexgroup = Mexdata.groupby(['Year','Month']).count()['Sentiment']

    Mexsentscores = []
    for i in Mexgroup:
        Mexsentscores.append(i)
    Mexsentscores.pop()
    # print(sum(Mexsentscores[0:7]))
    # print(sum(Mexsentscores[8:15]))

    Dates = ['2019 07','2019 08','2019 09','2019 10','2019 12','2020 01','2020 02','2020 03','2020 04','2020 05','2020 06','2020 07','2020 08','2020 09']
    plt.plot(Dates,USsentscores)
    plt.plot(Dates,Cansentscores)
    plt.plot(Dates,Mexsentscores)
    plt.title('Sentiment Score by Country')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    red_patch = mpatches.Patch(color='orange',label='Canada')
    blue_patch = mpatches.Patch(color='blue',label='USA')
    green_patch = mpatches.Patch(color='green',label = 'Mexico')
    plt.legend(handles=[red_patch,blue_patch,green_patch])
    plt.show()

def countryemotion():
    USfulldata = pd.DataFrame(columns=['TweetID','Date','TweetMessage'])

    USurl = f'Datasets/States/US/fullUS.csv'
    US_path = Path(USurl).resolve()
    USdata = pd.read_csv(US_path)
    print(len(USdata))
    time = "after"
    emotion = "fear"
    countries = ["USA","CAN","MEX"]

    if time == "before":
        months19 = ['07','08','09','10','12']
        months20 = ['01','02']
        year = '2019'
        for i in months19:
            USurl = f'Datasets/States/US/US{year}_{i}.csv'
            US_path = Path(USurl).resolve()
            USdata = pd.read_csv(US_path)
            USfulldata = USfulldata.append(USdata,sort=False)

    elif time == "after":
        months20 = ['03','04','05','06','07','08','09']

    year = '2020'
    for i in months20:
        USurl = f'Datasets/States/US/US{year}_{i}.csv'
        US_path = Path(USurl).resolve()
        USdata = pd.read_csv(US_path)
        USfulldata = USfulldata.append(USdata,sort=False)

    # USfiltered = USfulldata.query(f'emotion == "{emotion}"')
    USgroup = USfulldata.groupby(['emotion']).count()
    USfiltered = USgroup.query(f'emotion == "{emotion}"')

    USval = (USfiltered['joy'][0])

    Canurl = f'Datasets/Canada/AllCanada.csv'
    Can_path = Path(Canurl).resolve()
    Candata = pd.read_csv(Can_path)
    Canfiltered = Candata.query(f'emotion == "{emotion}"')

    Cangroup = Canfiltered.groupby(['Year','Month']).count()['emotion']
    Cansentscores = []
    for i in Cangroup:
        Cansentscores.append(i)
    Canval = sum(Cansentscores[8:15])

    Mexurl = f'Datasets/AllMexico.csv'
    Mex_path = Path(Mexurl).resolve()
    Mexdata = pd.read_csv(Mex_path)
    Mexfiltered = Mexdata.query(f'emotion == "{emotion}"')
    Mexgroup = Mexfiltered.groupby(['Year','Month']).count()['emotion']
    Mexsentscores = []
    for i in Mexgroup:
        Mexsentscores.append(i)
    MXval = sum(Mexsentscores[8:15])

    countryvals = [USval, Canval, MXval]
    print(countryvals)
    fig = go.Figure(data=go.Choropleth(
        locations = countries,
        z = countryvals,
        # text = df['COUNTRY'],
        colorscale = 'Oranges',
        autocolorscale=False,
        zmin = 0,
        zmax = 250000,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        # colorbar_tickprefix = '$',
        colorbar_title = f'Amount of {emotion} tweets',
    ))

    fig.update_layout(
        title_text=f'North America {emotion} {time} Covid-19',
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),
        # annotations = [dict(
        #     x=0.55,
        #     y=0.1,
        #     xref='paper',
        #     yref='paper',
        #     text='Source: <a href="https://www.cia.gov/library/publications/the-world-factbook/fields/2195.html">\
        #         CIA World Factbook</a>',
        #     showarrow = False
        # )]
    )

    fig.show()



if __name__ == "__main__":
    main()
