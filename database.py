import psycopg

class Database:
  def __init__(self, db_configs):
    self.db_configs = db_configs

  def execute(self, query, values=None):
    with psycopg.connect(**self.db_configs) as conn:
      with conn.cursor() as cur:
        cur.execute(query, values)

        if cur.description is not None:
          return cur.fetchall()

        conn.commit()
        return None