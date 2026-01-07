import os
from peewee import *
from datetime import datetime, timezone
import common.constants as CONST

db_path = os.path.join(CONST.ROOT_DIR, "output", "eval_runs.db")
if os.path.exists(db_path):
    print(f"Using existing database at {db_path}")
else:
    print(f"Creating new database at {db_path}")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)


db = SqliteDatabase(db_path)


# For PostgreSQL
# db = PostgresqlDatabase(
#     'your_db_name',
#     user='your_user',
#     password='your_password',
#     host='your_host',
#     port=5432
# )

class BaseModel(Model):
    class Meta:
        database = db

class Miner(BaseModel):
    hotkey = CharField(unique=True)
    created_at = DateTimeField(default=datetime.now(timezone.utc))

class Evaluation(BaseModel):
    run_id = CharField(null=True)
    miner = ForeignKeyField(Miner, backref='evaluations')
    created_at = DateTimeField(default=datetime.now(timezone.utc))
    eval_name = CharField()
    model_name = CharField()
    provider_name = CharField(default="unknown")
    score = FloatField()
    success = BooleanField()
    temperature = FloatField(default=0.0)
    duration_seconds = FloatField()
    rows_evaluated = IntegerField(default=0)
    comments = TextField(null=True) 
    
class MinerResponse(BaseModel):

    class Meta:
        table_name = 'miner_responses'

    run_id = CharField(null=True)    
    miner = ForeignKeyField(Miner, backref='responses')
    hotkey = CharField()
    created_at = DateTimeField(default=datetime.now(timezone.utc))
    query = TextField()
    num_recs = IntegerField()
    response = TextField()
    model_name = CharField()
    provider_name = CharField(default="unknown")
    temperature = FloatField(default=0.0)
    duration_seconds = FloatField()
   

    