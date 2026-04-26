"""
Re-exports the most-used DB lifecycle functions and dependencies so
app/main.py and app/api/deps.py can import from a single location.
"""

from app.db.postgres import (  # noqa: F401
    close_db_pool,
    create_db_pool,
    get_db,
    get_db_context,
    get_engine,
)
from app.db.redis import (  # noqa: F401
    close_redis_pool,
    create_redis_pool,
    get_redis,
    is_token_revoked,
    revoke_token,
)
from app.db.timescale import (  # noqa: F401
    close_timescale_pool,
    create_timescale_pool,
    get_quality_trend,
    get_rom_series,
    get_session_frequency,
    write_metric_batch,
)
from app.db.s3 import (  # noqa: F401
    delete_video,
    delete_videos_batch,
    download_video,
    generate_presigned_download_url,
    generate_presigned_upload_url,
    intake_key,
    object_exists,
    session_recording_key,
)