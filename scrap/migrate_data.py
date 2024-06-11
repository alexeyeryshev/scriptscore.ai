import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import os

def export_sqlite_to_csv(sqlite_db, tables):
    # Connect to SQLite database
    sqlite_conn = sqlite3.connect(sqlite_db)
    
    # Export tables to CSV
    for table in tables:
        df = pd.read_sql_query(f'SELECT * FROM {table}', sqlite_conn)
        df.to_csv(f'{table}.csv', index=False)
    
    # Close the connection
    sqlite_conn.close()

def import_csv_to_mysql(mysql_connection_string, tables):
    # Connect to MySQL database
    mysql_engine = create_engine(mysql_connection_string)
    
    # Import tables from CSV to MySQL
    for table in tables:
        df = pd.read_csv(f'{table}.csv')
        df.to_sql(table, mysql_engine, if_exists='append', index=False)

    print("Data has been successfully imported into MySQL database.")

if __name__ == "__main__":
    sqlite_db = 'data.db'
    mysql_connection_string = os.getenv('MYSQL_STRING')
    tables = ['simulations', 'personas', 'reviews']
    
    # export_sqlite_to_csv(sqlite_db, tables)
    import_csv_to_mysql(mysql_connection_string, tables)
