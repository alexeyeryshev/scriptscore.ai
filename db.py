import json
from sqlalchemy import text

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
    cursor = con.execute(text("""
        INSERT INTO personas VALUES (NULL, :ageStart, :ageEnd, :gender, :ethnicity, :location, :profession, :education, :income)
    """), persona)
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
