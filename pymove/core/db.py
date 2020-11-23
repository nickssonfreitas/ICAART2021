import psycopg2
import io
import pandas as pd
import tempfile


def connect_database(db_name='tornozelados', usr='postgres', pswd='postgres', host='localhost', port='5432'):
    try:
        conn = psycopg2.connect(database = db_name, user=usr, password = pswd, host=host, port=port)
        print('user connected to database')
        return conn
    except Exception as e:
        raise e

##it is faster than read_sql_tmpfile
def read_sql_inmem_uncompressed(query, conn):
    copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
       query=query, head="HEADER")
    
    cur = conn.cursor()
    store = io.StringIO() # create object StringIO
    cur.copy_expert(copy_sql, store)
    store.seek(0) #  move the cursor over it data like seek(0) for start of file
    df = pd.read_csv(store)
    cur.close() # free memory in cursor
    store.close() # free memory in StringIO
    return df


def read_sql_tmpfile(query, conn):
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
        query=query, head="HEADER"
    )
    cur = conn.cursor()
    cur.copy_expert(copy_sql, tmpfile)
    tmpfile.seek(0)
    df = pd.read_csv(tmpfile)
    return df