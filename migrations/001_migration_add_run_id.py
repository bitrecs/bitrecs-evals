"""Peewee migrations -- 001_migration_add_run_id.py.

Some examples (model - class or model name)::

    > Model = migrator.orm['table_name']            # Return model in current state by name
    > Model = migrator.ModelClass                   # Return model in current state by name

    > migrator.sql(sql)                             # Run custom SQL
    > migrator.run(func, *args, **kwargs)           # Run python function with the given args
    > migrator.create_model(Model)                  # Create a model (could be used as decorator)
    > migrator.remove_model(model, cascade=True)    # Remove a model
    > migrator.add_fields(model, **fields)          # Add fields to a model
    > migrator.change_fields(model, **fields)       # Change fields
    > migrator.remove_fields(model, *field_names, cascade=True)
    > migrator.rename_field(model, old_field_name, new_field_name)
    > migrator.rename_table(model, new_table_name)
    > migrator.add_index(model, *col_names, unique=False)
    > migrator.add_not_null(model, *field_names)
    > migrator.add_default(model, field_name, default)
    > migrator.add_constraint(model, name, sql)
    > migrator.drop_index(model, *col_names)
    > migrator.drop_not_null(model, *field_names)
    > migrator.drop_constraints(model, *constraints)

"""

import sys
import os
from contextlib import suppress

# Add project root to Python path (use cwd since __file__ isn't available)
project_root = os.getcwd()  # Assumes pw_migrate is run from project root
sys.path.insert(0, project_root)

import peewee as pw
from peewee_migrate import Migrator

# Import your models (not strictly needed for raw SQL, but good practice)
from db.models.eval import Miner, Evaluation

with suppress(ImportError):
    import playhouse.postgres_ext as pw_pext


def migrate(migrator: Migrator, database, fake=False, **kwargs):
    """Add run_id column to Evaluation table using raw SQL."""
    # Use raw SQL to avoid model lookup issues
    migrator.sql("ALTER TABLE evaluation ADD COLUMN run_id VARCHAR(255) NULL;")


def rollback(migrator: Migrator, database, fake=False, **kwargs):
    """Rollback: Drop run_id column using raw SQL."""
    migrator.sql("ALTER TABLE evaluation DROP COLUMN run_id;")

