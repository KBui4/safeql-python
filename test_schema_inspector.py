"""Run with: python -m unittest test_schema_inspector -v"""

import unittest

from schema_inspector import ALL_COLUMNS_QUERY, COLUMNS_QUERY, TABLES_QUERY, SchemaInspector


class FakeDb:
  def __init__(self, responses):
    self.responses = responses
    self.calls = []

  def execute(self, query, values=None):
    self.calls.append((query, values))
    if isinstance(query, str) and query in self.responses:
      return self.responses[query]
    return self.responses["values"]


USERS = ("users", "id", "integer", "NO", None, None)
USERS_NAME = ("users", "name", "character varying", "YES", None, 50)
POSTS = ("posts", "txt", "text", "YES", None, None)


class SchemaInspectorTests(unittest.TestCase):
  def test_get_tables(self):
    db = FakeDb({TABLES_QUERY: [("posts",), ("users",)]})
    self.assertEqual(SchemaInspector(db).get_tables(), ["posts", "users"])
    self.assertEqual(db.calls[0][1], ("public",))

  def test_get_columns(self):
    db = FakeDb({COLUMNS_QUERY: [USERS[1:], USERS_NAME[1:]]})
    columns = SchemaInspector(db).get_columns("users")
    self.assertEqual(db.calls[0][1], ("public", "users"))
    self.assertEqual(columns[0], {"name": "id", "data_type": "integer", "nullable": False, "default": None, "max_length": None})
    self.assertEqual(columns[1]["max_length"], 50)
    self.assertTrue(columns[1]["nullable"])

  def test_get_schema_keeps_tables_without_columns(self):
    db = FakeDb({
      TABLES_QUERY: [("empty",), ("posts",), ("users",)],
      ALL_COLUMNS_QUERY: [POSTS, USERS, USERS_NAME],
    })
    schema = SchemaInspector(db).get_schema()
    self.assertEqual(list(schema), ["empty", "posts", "users"])
    self.assertEqual(schema["empty"], [])
    self.assertEqual([c["name"] for c in schema["users"]], ["id", "name"])

  def test_get_column_types(self):
    db = FakeDb({TABLES_QUERY: [("users",)], ALL_COLUMNS_QUERY: [USERS, USERS_NAME]})
    types = SchemaInspector(db).get_column_types()
    self.assertEqual(types, {("users", "id"): "integer", ("users", "name"): "character varying"})

  def test_get_values_quotes_identifiers(self):
    db = FakeDb({"values": [("Hana",), ("Leo",)]})
    values = SchemaInspector(db).get_values("users", 'na"me', limit=5)
    self.assertEqual(values, ["Hana", "Leo"])
    query, params = db.calls[0]
    self.assertEqual(params, (5,))
    rendered = repr(query)
    self.assertIn("'public'", rendered)
    self.assertIn("'users'", rendered)
    self.assertIn("'na\"me'", rendered)


if __name__ == "__main__":
  unittest.main()
