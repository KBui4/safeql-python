from configs import get_db_env
from database import Database

def main():
  configs = get_db_env()
  db = Database(configs)

  # Here's a few test queries
  create_test = """
  CREATE TABLE IF NOT EXISTS test
  ( id INTEGER NOT NULL,
    words TEXT,
    PRIMARY KEY(id)
  );
  """

  insert_query = """
  INSERT INTO test
  VALUES(%s, %s),
  (%s, %s),
  (%s, %s)
  ON CONFLICT (id) 
  DO NOTHING;
  """
  insert_values = (1, "testing insert", 2, "another example", 3, "last one")

  select_query = """
  SELECT *
  FROM test
  """

  db.execute(create_test)
  db.execute(insert_query, insert_values)
  results = db.execute(select_query)
  print(results)

  return

if __name__ == "__main__":
  main()