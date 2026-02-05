#!/bin/bash
set -e

# Wait for any initialization if needed
echo "HR Assessment Pipeline - Docker Container"
echo "=========================================="
echo ""
echo "Available commands:"
echo "  - Process single file:    python main.py audio/file.wav"
echo "  - Process folder:         python main.py audio/folder/ --limit 10"
echo "  - Process team:           python process_team_recordings.py 'Team Recordings' --person Name"
echo "  - Start API:              uvicorn api:app --host 0.0.0.0 --port 8000"
echo ""

# Execute the command passed to docker run
exec "$@"
