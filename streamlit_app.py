from langchain_openai import ChatOpenAI
import streamlit as st
import altair as alt
from openai import OpenAI
import pandas as pd

from simulation import simulate
from db import bootstrap_db


# Page title
st.set_page_config(page_title='ScriptScore.AI', page_icon='🎥')
st.title('🤖 Predictive Audience Intelligence')

# Meta
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai_client = OpenAI(api_key=OPENAI_API_KEY)
if st.secrets["local"] == "true":
    conn = st.connection("sqlite")
    bootstrap_db(conn, local=True)
else:
    conn = st.connection("mysqldo")
    bootstrap_db(conn, local=False)

with st.expander('About this plartform', expanded=False):
  st.markdown('**What can this app do?**')
  st.info('The ScriptScore platform enables entertainment industry professionals to utilize AI for predicting audience sentiment during the initial stages of content creation.')

  st.markdown('**How to use the app?**')
  st.markdown('''
    * **Provide an Idea of a Plot**: Input a brief description of your plot idea, including the story summary, key themes, or specific scenes.
    * **Choose Your Dream Cast**: Select the actors and actresses you wish to cast in your project.
    * **Speculate on a Budget**: Estimate the budget, including costs for production, talent, special effects, and other relevant expenses.
    * **Let AI Work Its Magic**: Allow the AI to analyze the data and predict audience sentiment, providing valuable insights at the earliest stages of content creation.
  ''')
  st.warning('Unfortunately, this app runs on a commodity server and may not be able to handle large-scale simulations. Please be patient and considerate of other users.')

previous_simulations = conn.query('SELECT * FROM simulations ORDER BY id DESC LIMIT 5', ttl=1)
if len(previous_simulations):
    st.subheader('📽️ Previous Projects')

    last_simulations_len = min(len(previous_simulations), 5)
    for i, col in enumerate(st.columns(last_simulations_len)):
        with col:
            simulation = previous_simulations.iloc[i]
            def on_click(simulation_id):
                st.session_state['simulation'] = simulation_id
            simulation_id = simulation['id'].item()
            st.button(simulation["name"], on_click=on_click, key=simulation_id, kwargs={"simulation_id": simulation_id})

if simulation_id := st.session_state.get('simulation'):
    simulation = conn.query(f'SELECT * FROM simulations WHERE id = {simulation_id}')
    reviews = conn.query(f'''
                          SELECT 
                            reviews.id, 
                            reviews.simulation, 
                            reviews.persona, 
                            reviews.review, 
                            reviews.rating, 
                            reviews.lookingForward,
                            personas.ageStart,
                            personas.ageEnd,
                            personas.gender,
                            personas.ethnicity,
                            personas.location,
                            personas.profession,
                            personas.education,
                            personas.income
                          FROM reviews 
                          LEFT JOIN personas ON reviews.persona = personas.id WHERE simulation = {simulation_id}
                          ''')
    reviews['ageRange'] = reviews['ageStart'].astype(str) + '-' + reviews['ageEnd'].astype(str)
    locations2Lat = { 
        "North America" : 37.0902,
        "Europe": 55.3781,
        "Asia": 34.0479,
        "South America": -14.2350,
        "Africa": -8.7832,
        "Oceania": -25.2744
    }
    locations2Lon = {
        "North America" : -95.7129,
        "Europe": -3.4360,
        "Asia": 100.6197,
        "South America": -51.9253,
        "Africa": 34.5085,
        "Oceania": 133.7751
    }
    reviews['lat'] = reviews['location'].apply(lambda x: locations2Lat.get(x, 0))
    reviews['lon'] = reviews['location'].apply(lambda x: locations2Lon.get(x, 0))

    st.subheader("", divider='rainbow')
    st.subheader(simulation['name'].item())
    poster = simulation['poster'].item()
    if poster:
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            st.image(simulation['poster'].item(), use_column_width=True)
        with col2:
            st.write(simulation['synopsis'].item())
    else:
        st.write(simulation['synopsis'].item())

    cast = simulation['cast'].item().split(',')
    if len(cast) and cast[0] != '':
        st.subheader('🎭 Cast')
        st.write(', '.join(cast))

    # st.map(reviews, size=200000)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('⭐ Average Rating')
        st.metric('Average Rating', f'{reviews["rating"].mean():.2f} / 5', label_visibility='collapsed')
    with col2:
        st.subheader('👍 Looking forward?')
        total = reviews['lookingForward'].sum()
        count = reviews['lookingForward'].count()
        st.metric('Looking forward', f"{total/count * 100}%", label_visibility='collapsed')

    def breakdown_char(dim, title):
        age_rating = reviews.groupby(dim)['rating'].mean().reset_index()
        bars = alt.Chart(age_rating).mark_bar(
            cornerRadiusTopLeft=10,
            cornerRadiusTopRight=10,
            color='#F63366'
        ).encode(
            x=alt.X(dim, sort=None, title='', ),
            y=alt.Y('rating', title=''),
            tooltip=[alt.Tooltip(dim, title=title), alt.Tooltip('rating', title='Average Rating')]
        ).properties(
            height=200
        ).configure_view(strokeOpacity=0).configure_axis(
            grid=False,
            labelAngle=0
        )
        chart = bars 

        st.altair_chart(chart, use_container_width=True)
    

    st.subheader('📊 Sentiment Breakdown')
    st.markdown('**Age**')
    breakdown_char("ageRange", "Age Range")
    # breakdown_char("ethnicity", "Ethnicity")
    st.markdown('**Ethinicity**')
    st.dataframe(reviews.groupby('ethnicity')['rating'].mean().reset_index(), hide_index=True, use_container_width=True, column_config={"ethnicity": "Ethnicity", "rating": 
                                                                                                                                        st.column_config.ProgressColumn("Average Rating", min_value=0, max_value=5, format="%d")})
    st.markdown('**Gender**')
    breakdown_char("gender", "Gender")
    st.markdown('**Profession**')
    # breakdown_char("profession", "Profession")
    st.dataframe(reviews.groupby('profession')['rating'].mean().reset_index(), hide_index=True, use_container_width=True, column_config={"profession": "Profession", "rating": 
                                                                                                                                        st.column_config.ProgressColumn("Average Rating", min_value=0, max_value=5, format="%d")})
    st.markdown('**Education**')
    breakdown_char("education", "Education")
    st.markdown('**Income**')
    breakdown_char("income", "Income")


# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.header('ScriptScore.AI', divider='rainbow')

    st.header('Title')
    title = st.text_input('Title', 'My amazing movie', label_visibility='collapsed')
    st.header('Type')
    content_type = st.selectbox('Type', ['Movie', 'TV Show', 'Documentary', 'Web Series'], label_visibility='collapsed', placeholder='Select the type of content')
    st.header('Genres')
    genres = st.multiselect('Genres', ['Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller'], label_visibility='collapsed', placeholder='Select the genres')
    st.header('Script')
    script = st.text_area('Script', 'Write your script here', label_visibility='collapsed')
    st.header('Cast')
    cast = st.multiselect('Cast', ['Tom Cruise', 'Tom Hanks', 'Brad Pitt', 'Leonardo DiCaprio', 'Will Smith', 'Scarlett Johansson', 'Angelina Jolie', 'Jennifer Aniston', 'Julia Roberts', 'Meryl Streep'], label_visibility='collapsed', placeholder='Select your dream cast')
    st.header('Budget')
    budget = st.slider('Budget', 0, 500, 25, 1, format='$%dM', label_visibility='collapsed' )

    model = None
    with st.expander('Advanced configuraiton', expanded=False):
        skip_poster_generation = st.checkbox('Skip poster generation', value=False)
        audience_size = st.number_input('Audience size', min_value=1, max_value=30, value=10)
        model = st.selectbox('Model', ['GPT-3.5', 'GPT-4.0'], index=0)
        temperature = st.slider('Temperature', 0.0, 1.0, 0.2, 0.1)

    def run_simulation(model, openai_client):
        progress = st.progress(0)
        content = {
            "name": title,
            "type": content_type,
            "genres": genres,
            "script": script,
            "cast": cast,
            "budget": budget
        }
        openai_model_35 = ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo-1106", openai_api_key=OPENAI_API_KEY, model_kwargs={"response_format": {"type": "json_object"}},)
        openai_model_4 = ChatOpenAI(temperature=temperature, model_name="gpt-4o", openai_api_key=OPENAI_API_KEY, model_kwargs={"response_format": {"type": "json_object"}},)
        model = openai_model_35 if model == 'GPT-3.5' else openai_model_4
        with conn.session as session:
            simulation_id = simulate(model, openai_client, session, content, lambda x: progress.progress(x), {"how_many": audience_size, "skip_poster_generation": skip_poster_generation})
        st.session_state['simulation'] = simulation_id
        progress.empty()

    st.button('🪄', use_container_width=True, on_click=run_simulation, kwargs={"model": model, "openai_client": openai_client}, type="primary")

    #Debug only
    with st.expander('Debug information', expanded=False):
        st.write(f'🔍 Simulation: {simulation_id}')