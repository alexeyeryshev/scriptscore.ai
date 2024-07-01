import json
from sqlalchemy import Column, Float, ForeignKey, Integer, LargeBinary, MetaData, String, Table, Text, text
from streamlit.connections import SQLConnection
from sqlalchemy.dialects.mysql import MEDIUMBLOB

def create_tables(engine):
    meta = MetaData()
    
    simulations = Table(
        'simulations', meta,
        Column('id', Integer, primary_key=True, autoincrement=True, nullable=False),
        Column('name', String(255)),
        Column('type', String(255)),
        Column('genres', String(1024)),
        Column('cast', String(4096)),
        Column('script', Text),
        Column('budget', Float(10,2)),
        Column('synopsis', Text),
        Column('poster', MEDIUMBLOB)
    )
    
    personas = Table(
        'personas', meta,
        Column('id', Integer, primary_key=True, autoincrement=True, nullable=False),
        Column('ageStart', Integer),
        Column('ageEnd', Integer),
        Column('gender', String(50)),
        Column('ethnicity', String(255)),
        Column('location', String(255)),
        Column('profession', String(255)),
        Column('education', String(255)),
        Column('income', String(255))
    )
    
    reviews = Table(
        'reviews', meta,
        Column('id', Integer, primary_key=True, autoincrement=True, nullable=False),
        Column('simulation', Integer, ForeignKey('simulations.id'), nullable=False),
        Column('persona', Integer, ForeignKey('personas.id'), nullable=False),
        Column('source', Text, nullable=False),
        Column('review', Text),
        Column('rating', Float(3,2)),
        Column('lookingForward', Integer)
    )
    
    meta.create_all(engine)

def bootstrap_db(conn: SQLConnection, local = True):
    if local:
        with conn.session as session:
            session.execute(text('''
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                name TEXT,
                type TEXT,
                genres TEXT,
                script TEXT,
                cast TEXT,
                budget REAL,
                synopsis TEXT,
                poster BLOB
            )
            '''))

            session.execute(text('''
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

            session.execute(text('''
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
    else:
        create_tables(conn.engine)


def create_simulation_in_db(session, content, synopsis, poster):
    simulation = {
        **content,
        "genres": ",".join(content["genres"]),
        "cast": ",".join(content["cast"]),
        "script": content["script"],
        "synopsis": synopsis["synopsis"],
        "poster": poster, 
    }
    cursor = session.execute(text("INSERT INTO simulations VALUES(NULL, :name, :type, :genres, :cast, :script, :budget, :synopsis, :poster);"), simulation)
    return cursor.lastrowid

def create_persona_in_db(session, persona):
    cursor = session.execute(text("""
        INSERT INTO personas VALUES (NULL, :ageStart, :ageEnd, :gender, :ethnicity, :location, :profession, :education, :income)
    """), persona)
    return cursor.lastrowid

def create_review_in_db(session, simulation, persona, review):
    data = {
        "simulation": simulation,
        "persona": persona,
        "source": json.dumps(review),
        "review": review["review"],
        "rating": review["rating"],
        "lookingForward": review["lookingForward"],
    }
    cursor = session.execute(text("INSERT INTO reviews VALUES(NULL, :simulation, :persona, :source, :review, :rating, :lookingForward)"), data)
    return cursor.lastrowid