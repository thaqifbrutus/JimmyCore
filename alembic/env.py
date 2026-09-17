from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

from app.config import DATABASE_URL
from db.database import Base

# Import every model module so Base.metadata actually knows about all
# tables — SQLAlchemy only registers a model with Base once its module has
# been imported somewhere. Missing one of these silently drops that table
# from autogenerate's view of "the real schema."
from app.models import dataset, report, catalog_dataset, audit_log, gov_data_cache  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Single source of truth for the DB URL — same env var the app itself
# reads, so migrations can never point somewhere the app doesn't.
config.set_main_option("sqlalchemy.url", DATABASE_URL or "")

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Generates the SQL a migration WOULD run, without connecting to a real
    database. Useful for reviewing exactly what a migration does before
    running it for real: `alembic upgrade head --sql`.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
