
import pyodbc
import pandas as pd

SERVER = 'Local_Host'
DATABASE = 'YouTubeSentimentAnalyzer'

connection_string = f'''
DRIVER={{SQL Server}};
SERVER={SERVER};
DATABASE={DATABASE};
Trusted_Connection=yes;
'''

def save_analysis(
    video_id,
    video_title,
    positive_percent,
    negative_percent,
    neutral_percent,
    total_comments
):
    conn = pyodbc.connect(connection_string)

    cursor = conn.cursor()

    query = """
    INSERT INTO analyses (
        video_id,
        video_title,
        positive_percent,
        negative_percent,
        neutral_percent,
        total_comments
    )
    VALUES (?, ?, ?, ?, ?, ?)
    """

    cursor.execute(query, (
        video_id,
        video_title,
        positive_percent,
        negative_percent,
        neutral_percent,
        total_comments
    ))

    conn.commit()

    print("✅ DATA SAVED")

    conn.close()

def get_all_analyses():

    conn = pyodbc.connect(connection_string)

    query = """
    SELECT *
    FROM analyses
    ORDER BY created_at DESC
    """

    df = pd.read_sql(query, conn)

    conn.close()

    return df
