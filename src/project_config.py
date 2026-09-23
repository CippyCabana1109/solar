"""Shared analysis constants used across dissertation pipeline stages."""

DAYTIME_START_HOUR = 6
DAYTIME_END_HOUR_EXCLUSIVE = 19
DAYTIME_DEFINITION = "06:00-18:59"

# Monthly expanding origins; each origin evaluates the following seven days.
VALIDATION_ORIGINS = (
    "2024-05-01",
    "2024-06-01",
    "2024-07-01",
    "2024-08-01",
    "2024-09-01",
    "2024-10-01",
    "2024-11-01",
    "2024-12-01",
    "2025-01-01",
)
VALIDATION_WINDOW_DAYS = 7