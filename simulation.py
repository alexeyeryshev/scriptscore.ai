import json
import random
from sqlalchemy import text

from langchain_core.prompts import PromptTemplate
import requests


file_path = 'demography.json'
demography = None
with open(file_path, 'r') as file:
    demography = json.load(file)

individual_prompt = PromptTemplate.from_template("""
You're an individual with a specific age, gender, or ethnic background from a defined geographic location.

Those parameters are specified within the following block.
It starts with "BEGIN demography" and ends with "END demography".

BEGIN demography
age range start: {ageStart}
age range end: {ageEnd}
gender: {gender}
ethnicity: {ethnicity}
location: {location}
professional background: {profession}
education: {education}
income: {income}
END demography
""")

simple_review_prompt = PromptTemplate.from_template("""
Write a review about Witcher season 2? 
""")

project_prompt = PromptTemplate.from_template("""
Below are high-level movie or TV show ideas and some essential parameters.
It starts with "BEGIN content" and ends with "END content".

BEGIN content
name: {name}
type: {type}
cast: {cast}
budget: {budget}
END content
""")

cta_prompt = PromptTemplate.from_template("""
Write a simple review as someone who just watched the {name}.
Highlight points that you liked and didn't like.
Provide a rating that you give to this content on a scale from 1 to 5.
If you look forward to watch it again answer with 0 for No or 1 for Yes.
Be honest, your feedback is valuable and will be applied by a studio to improve the content.

Answer in JSON format, use following structure:
{{
    "review": <your answer>,
    "rating": <your rating>,
    "lookingForward": <your answer>
}}
""")

synopsis_cta_prompt = PromptTemplate.from_template("""
Generate synopsis to put on a website or a dvd box for this content.
Print it out in JSON format with synopsis key.
""")

poster_generation_prompt = PromptTemplate.from_template("""
Generate a poster for a cinema content with a synopsis below.
It's important - do not put text on it at all.

BEGIN synopsis
{synopsis}
END synopsis
""")

full_prompt = individual_prompt + project_prompt + cta_prompt
synopsis_prompt = project_prompt + synopsis_cta_prompt

def get_random_persona(demography, seed=None):
    if seed:
        random.seed(seed)
    while True:
        permutation = {}
        permutation['age'] = random.choice(demography['age'])
        permutation['gender'] = random.choice(demography['genders'])
        permutation['ethnicity'] = random.choice(demography['ethnicities'])
        permutation['location'] = random.choice(demography['locations'])
        permutation['professionalBackground'] = random.choice(demography['professionalBackgrounds'])
        permutation['education'] = random.choice(demography['educations'])
        permutation['incomeLevel'] = random.choice(demography['incomeLevels'])
        yield permutation

def generate_demography_prompt_input(persona):
    return {
    "ageStart": persona["age"]["start"],
    "ageEnd": persona["age"]["end"],
    "gender": persona["gender"],
    "ethnicity": persona["ethnicity"],
    "location": persona["location"],
    "profession": persona["professionalBackground"],
    "education": persona["education"],
    "income": persona["incomeLevel"]
    }

def generate_content_prompt_input(content):
    return {
        "name": content["name"],
        "type": content["type"],
        "cast": ",".join(content["cast"]),
        "budget": content["budget"]
    }

# prompt generator
def generate_prompt_input(persona, content):
    return {
        **generate_demography_prompt_input(persona),
        **generate_content_prompt_input(content)
    }

def bootstrap_db(con):
    with con:
        con.execute(text('''
        CREATE TABLE IF NOT EXISTS simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT,
            type TEXT,
            cast TEXT,
            budget REAL,
            synopsis TEXT,
            poster BLOB
        )
        '''))

        con.execute(text('''
        CREATE TABLE IF NOT EXISTS personas (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            ageStart INTEGER,
            ageEnd INTEGER,
            gender TEXT,
            ethnicity TEXT,
            location TEXT,
            profession TEXT,
            education TEXT,
            income TEXT
        )
        '''))

        con.execute(text('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            simulation INTEGER NOT NULL,
            persona INTEGER NOT NULL,
            source TEXT NOT NULL,
            review TEXT,
            rating REAL,
            lookingForward INTEGER,
            
            FOREIGN KEY (simulation) REFERENCES simulations(ID)
            FOREIGN KEY (persona) REFERENCES personas(ID)
        )
        '''))

        con.commit()

def create_simulation_in_db(con, content, synopsis, poster):
    simulation = {
        **content,
        "cast": ",".join(content["cast"]),
        "synopsis": synopsis["synopsis"],
        "poster": poster, 
    }
    cursor = con.execute(text("INSERT INTO simulations VALUES(NULL, :name, :type, :cast, :budget, :synopsis, :poster);"), simulation)
    return cursor.lastrowid

def create_persona_in_db(con, persona):
    data = generate_demography_prompt_input(persona)
    cursor = con.execute(text("""
        INSERT INTO personas VALUES (NULL, :ageStart, :ageEnd, :gender, :ethnicity, :location, :profession, :education, :income)
    """), data)
    return cursor.lastrowid

def create_review_in_db(con, simulation, persona, review):
    data = {
        "simulation": simulation,
        "persona": persona,
        "source": json.dumps(review),
        "review": review["review"],
        "rating": review["rating"],
        "lookingForward": review["lookingForward"],
    }
    cursor = con.execute(text("INSERT INTO reviews VALUES(NULL, :simulation, :persona, :source, :review, :rating, :lookingForward)"), data)
    return cursor.lastrowid

def generate_poster(openai_client, synopsis):
    post_generation_prompt_resolved = poster_generation_prompt.invoke(synopsis["synopsis"])
    image_response = openai_client.images.generate(
        model="dall-e-3",
        prompt = post_generation_prompt_resolved.json(),
        size = "1024x1024",
        quality="standard",
        n=1,
    )
    image_url = image_response.data[0].url
    print(image_url)
    response = requests.get(image_url)
    if response.status_code == 200:
        response = response.content
    else:
        raise Exception("Failed to generate poster")

    return response

def simulate(model, openai_client, con, content, on_simulate_tick, config):
    random_persona_generator = get_random_persona(demography, 42)
    review_chain = full_prompt | model
    synopsis_chain = synopsis_prompt | model
    how_many = config["how_many"]
    skip_poster_generation = config["skip_poster_generation"]

    with con:
        persona = next(random_persona_generator)
        ai_message = synopsis_chain.invoke(generate_prompt_input(persona, content))
        synopsis = json.loads(ai_message.content)
        on_simulate_tick(1 / (how_many + 2))

        poster_bytes = None
        if not skip_poster_generation:
            poster_bytes = generate_poster(openai_client, synopsis)
            on_simulate_tick(2 / (how_many + 2))

        simulation_id = create_simulation_in_db(con, content, synopsis, poster_bytes)
        review_ids = []
        
        for i in range(how_many):
            persona = next(random_persona_generator)
            print(persona)
            persona_id = create_persona_in_db(con, persona)
            prompt_input = generate_prompt_input(persona, content)
            ai_message = review_chain.invoke(prompt_input)
            review = json.loads(ai_message.content)
            print(review, json.dumps(review))

            review_id = create_review_in_db(con, simulation_id, persona_id, review)
            review_ids.append(review_id)
            on_simulate_tick((i + 2) / (how_many + 2))
        
        con.commit()
    
    return simulation_id