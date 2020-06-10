import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px

DATA_URL = (r"C:\Users\Mitsy\Projects\fuzzy-potato\Motor_Vehicle_Collisions_Crashes.csv")
st.title("Motor Vehicle Collisions in New York City")
st.markdown("This is a Streamlit dashboard that can be used"
"to analyze motor vehicle collision in NYC")

@st.cache(persist=True)
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows,parse_dates=[['CRASH DATE','CRASH TIME']])
    #no missing values
    data.dropna(subset=['LATITUDE','LONGITUDE'],inplace=True)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase,axis='columns',inplace=True)
    data.rename(columns={'crash date_crash time':'date/time','number of persons injured':'injured_people',"number of pedestrians injured":'injured_pedestrians','number of cyclist injured':'injured_cyclists','number of motorist injured':'injured_motorists'},inplace=True)
    return data

df = load_data(100000)
org_data = df
st.header("Where are the most people injured in NYC?")
injured_ppl = st.slider("Number of persons injured in vehicle collisions",0,19)
st.map(df.query("injured_people >= @injured_ppl")[['latitude','longitude']].dropna(how="any"))

st.header("How many collisions occur during an hour in the day")
hour = st.sidebar.slider("Hour to look at", 0,23)
df = df[df['date/time'].dt.hour == hour]
###
st.markdown("Vehicle Collisions between %i:00 and %i:00" % (hour,(hour +1)%24))
locs = (np.average(df['latitude']),np.average(df['longitude']))
st.write(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={"latitude":locs[0], "longitude":locs[1],
        "zoom":11,
        "pitch":50},
    layers=[pdk.Layer(
        "HexagonLayer",
        data=df[['date/time','latitude','longitude']],
        get_position=['longitude','latitude'],
        radius=100,
        extruded=True,
        pickable=True,
        elevation_scale=4,
        elevation_range=[0,1000],
    ),],
))
###
st.subheader("Breakdown of collisions by minute between %i:00 and %i:00" % (hour, (hour+1)%24))
filtered = df[
    (df['date/time'].dt.hour >= hour) & (df['date/time'].dt.hour < (hour+1))
]
hist = np.histogram(filtered['date/time'].dt.minute, bins=60,range=(0,60))[0]
chart_data = pd.DataFrame({'minute':range(60),'crashes':hist})
fig = px.bar(chart_data, x='minute',y='crashes',hover_data=['minute','crashes'],height=400)
st.write(fig)
###

st.header("Top 5 most dangerous streets by affected type")
sel = st.selectbox("Affected type of people",['Pedestrians','Cyclists','Motorists'])

if sel == "Pedestrians":
    st.write(org_data.query("injured_pedestrians >= 1")[["on street name","injured_pedestrians"]].sort_values(by=['injured_pedestrians'],ascending=False).dropna(how="any")[:5])
elif sel == "Cyclists":
    st.write(org_data.query("injured_cyclists >= 1")[["on street name","injured_cyclists"]].sort_values(by=['injured_cyclists'],ascending=False).dropna(how="any")[:5])
elif sel == "Motorists":
    st.write(org_data.query("injured_motorists >= 1")[["on street name","injured_motorists"]].sort_values(by=['injured_motorists'],ascending=False).dropna(how="any")[:5])



if st.checkbox("Show Raw Data",False):
    st.subheader('Raw Data')
    st.write(df)
