# look up which tables, columns and values exist in the database
# SafeQL uses these as the candidates when refining relations, attributes and values

from psycopg import sql

TABLES_QUERY = """
SELECT table_name
FROM information_schema.tables
WHERE table_schema = %s
  AND table_type = 'BASE TABLE'
ORDER BY table_name;
"""

COLUMNS_QUERY = """
SELECT column_name, data_type, is_nullable, column_default, character_maximum_length
FROM information_schema.columns
WHERE table_schema = %s
  AND table_name = %s
ORDER BY ordinal_position;
"""

ALL_COLUMNS_QUERY = """
SELECT c.table_name, c.column_name, c.data_type, c.is_nullable, c.column_default, c.character_maximum_length
FROM information_schema.columns c
JOIN information_schema.tables t
  ON t.table_schema = c.table_schema
 AND t.table_name = c.table_name
WHERE c.table_schema = %s
  AND t.table_type = 'BASE TABLE'
ORDER BY c.table_name, c.ordinal_position;
"""

def _column_dict(row):
  name, data_type, is_nullable, default, max_length = row
  return {
    "name": name,
    "data_type": data_type,
    "nullable": is_nullable == "YES",
    "default": default,
    "max_length": max_length,
  }

class SchemaInspector:
  def __init__(self, db, schema="public"):
    self.db = db
    self.schema = schema

  def get_tables(self): # Rels: every base table in the schema
    return [row[0] for row in self.db.execute(TABLES_QUERY, (self.schema,))]

  def get_columns(self, table): # Atts of one table, in the order they were defined
    rows = self.db.execute(COLUMNS_QUERY, (self.schema, table))
    return [_column_dict(row) for row in rows]

  def get_schema(self): # {table: [columns]} for the whole schema in one query
    schema = {table: [] for table in self.get_tables()}
    for row in self.db.execute(ALL_COLUMNS_QUERY, (self.schema,)):
      schema.setdefault(row[0], []).append(_column_dict(row[1:]))
    return schema

  def get_column_types(self): # {(table, column): data_type}, the type constraint used for type-based pruning
    return {
      (table, column["name"]): column["data_type"]
      for table, columns in self.get_schema().items()
      for column in columns
    }

  def get_values(self, table, column, limit=100): # Vals: distinct non-null values stored in a column
    # identifiers can't be passed as %s parameters, so quote them safely with psycopg.sql
    query = sql.SQL(
      "SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY {column} LIMIT %s"
    ).format(
      column=sql.Identifier(column),
      table=sql.Identifier(self.schema, table),
    )
    return [row[0] for row in self.db.execute(query, (limit,))]
