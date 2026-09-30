from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import DATABASE_URL


def _normalize_db_url(url: str) -> str:
    """
    Neon's pooled connection string (with `-pooler` in the hostname) runs
    through PgBouncer in transaction mode, which does not support
    channel_binding=require. Strip it if present — otherwise connections
    fail with SSL errors. Also normalizes the legacy `postgres://` scheme
    to `postgresql://` for SQLAlchemy 2.0.
    """
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    url = url.replace("channel_binding=require", "channel_binding=disable")
    # Clean up any resulting && or trailing ?&
    url = url.replace("&&", "&").rstrip("?&")
    return url


_db_url = _normalize_db_url(DATABASE_URL)

engine = create_engine(
    _db_url,
    # Recover from Neon's compute-suspend disconnects: ping before every
    # checkout, and force-recycle connections before Neon's ~5-minute
    # idle timeout. Without these, a stale pooled connection produces
    # "SSL connection has been closed unexpectedly".
    pool_pre_ping=True,
    pool_recycle=240,
    pool_size=5,
    max_overflow=5,
    pool_timeout=30,
    connect_args={
        "connect_timeout": 10,
        # TCP keepalives so the underlying socket survives idle periods
        # without being killed by Render or an intermediary NAT.
        "keepalives": 1,
        "keepalives_idle": 60,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    },
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app.models import dataset, report, audit_log, catalog_dataset, gov_data_cache
    Base.metadata.create_all(bind=engine)