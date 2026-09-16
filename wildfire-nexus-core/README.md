# WILDFIRE NEXUS CORE

Wildfire detection and tracking system using NASA FIRMS thermal data.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy environment file
cp .env.example .env
# Edit .env and add your FIRMS_MAP_KEY (get from https://firms.modaps.eosdis.nasa.gov/api/)

# Run the system
make run

# Or with uvicorn for development
make dev
```

## Features

- **Thermal Point Detection**: Ingests real-time thermal anomaly data from NASA FIRMS
- **Event Tracking**: Aggregates individual detections into wildfire events with lifecycle
- **Context Enrichment**: Adds weather data, proximity to settlements, risk assessment
- **Deduplication**: Prevents alert fatigue by suppressing duplicate notifications
- **Risk Prioritization**: Classifies fires as critical/warning/info based on threat level
- **Web Map**: Leaflet-based visualization of active fire events
- **Telegram Alerts**: Real-time notifications for critical fires

## Architecture

```
INGEST (FIRS, Weather) → DETECT (Filter, Track, Prioritize) → ALERT (Telegram, API)
                                                    ↓
                                               SQLite DB
```

## Configuration

Edit `config/settings.yaml` for:
- Region bounding boxes
- Detection thresholds
- Alert parameters

Set environment variables in `.env`:
- `FIRMS_MAP_KEY`: NASA FIRMS API key
- `TELEGRAM_BOT_TOKEN`: Telegram bot token (optional)
- `TELEGRAM_CHAT_ID`: Telegram chat ID for alerts (optional)

## API Endpoints

- `GET /health` - System health check
- `GET /api/v1/events` - List active fire events
- `GET /api/v1/events/{id}` - Event details
- `GET /api/v1/stats` - Statistics

## Development

```bash
# Run tests
make test

# Run linter
make lint

# Seed settlement data
make seed
```

## License

MIT
