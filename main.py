from safeql import safeql
from configs import get_db_env
from database import Database
from error_handler import inspect_query
from schema_inspector import SchemaInspector


def main():
  configs = get_db_env()
  db = Database(configs)

  # Here's a few test queries
  create_test = """
  CREATE TABLE IF NOT EXISTS employees (
    id SERIAL PRIMARY KEY,
    name TEXT,
    department TEXT,
    salary INT
  );
  """

  insert_query = """
  INSERT INTO employees (name, department, salary) VALUES
    (%s, %s, %s),
    (%s, %s, %s),
    (%s, %s, %s)
  ON CONFLICT DO NOTHING;
  """
  insert_values = ('Alice', 'Engineering', 180, 'Bob', 'Marketing', 150, 'Charlie', 'Engineering', 220)

  select_query = """
  SELECT *
  FROM employees
  """

  db.execute(create_test)
  db.execute(insert_query, insert_values)
  results = db.execute(select_query)

  inspector = SchemaInspector(db)

  # Running safeql
  wrong_query_example = "SELECT name, dept FROM employees WHERE salary > 160;" # dept not department
  print(safeql(wrong_query_example, db, inspector))
  return

if __name__ == "__main__":
  main()