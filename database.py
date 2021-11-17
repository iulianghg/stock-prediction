"""File containing the database class for interacting with the MySQL database.

Author: Maria Gospodinova
"""

import credentials
import pymysql

CREATE_DB_QUERY = "CREATE DATABASE IF NOT EXISTS twitter_stock_db"

CREATE_TABLE_QUERY = """
        CREATE TABLE IF NOT EXISTS twitter_stock_db.twitter_sent_tbl
        (id_str VARCHAR(255),
        created_at DATETIME,
        text VARCHAR(255),
        polarity INT
        )
        """

INSERT_QUERY = """INSERT INTO twitter_stock_db.twitter_sent_tbl
                                (id_str, created_at, text, polarity) VALUES
                                (%s, %s, %s, %s)"""

AVG_POL_QUERY = """SELECT AVG(polarity) \
        FROM twitter_stock_database.twitter_data_sentiment"""


class Database():
    """Initialises a database object and builds a cursor to execute queries."""

    def __init__(self):
        self.database = pymysql.connect(
                host=credentials.MYSQL_HOST,
                port=credentials.MYSQL_PORT,
                user=credentials.MYSQL_USER,
                password=credentials.MYSQL_PWD,
                charset=credentials.MYSQL_CHARSET,
                autocommit=credentials.MYSQL_AUTOCOMMIT
            )

        self.cursor = self.database.cursor()

    def cursor_execute(self, query: str) -> None:
        """Execute a query on the database.

        Arguments:
            query {str} -- MySQL query to execute
        """
        self.cursor.execute(query)

    def insert_tweet(self, database_columns: list) -> None:
        """Insert a tweet into the database.

        Arguments:
            database_columns {list} -- List of tweet attributes
        """
        self.cursor.execute(INSERT_QUERY, database_columns)

    def commit(self) -> None:
        """Commit changes to the database."""
        self.database.commit()
