import os
from dotenv import load_dotenv

class MissingEnvironmentVariable(Exception):
  pass

def get_db_env():
  load_dotenv()
  db_configs = {
    "dbname": os.getenv("DBNAME"),
    "user": os.getenv("USER"),
    "password": os.getenv("PASS"),
    "host": os.getenv("HOST"),
    "port": os.getenv("PORT")
  }

  if not all(db_configs.values()):
    missing_envs = [env for env, value in db_configs.items() if not value]
    raise MissingEnvironmentVariable(f"{missing_envs} are not defined.")

  return db_configs
