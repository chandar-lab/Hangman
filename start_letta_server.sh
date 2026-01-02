#!/bin/bash
#
# Letta Server v0.16.1 Startup Script
# For SLURM cluster with Conda + PostgreSQL
#

set -e

PROJECT_DIR="/home/mila/b/baldelld/scratch/hangman"
LETTA_ENV="$PROJECT_DIR/letta_server_env"
PG_DATA="$PROJECT_DIR/pg_data"
LETTA_REPO="$PROJECT_DIR/letta_repo"

echo "=== Letta Server v0.16.1 Startup ==="
echo ""

# Load Anaconda
echo "Loading anaconda module..."
module load anaconda/3

# Activate environment
echo "Activating Letta environment..."
conda activate "$LETTA_ENV"

# Set PostgreSQL connection
export LETTA_PG_URI="postgresql://letta:letta@localhost:5432/letta"

# Check if PostgreSQL is running
if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo "Starting PostgreSQL..."
    pg_ctl -D "$PG_DATA" -l "$PG_DATA/logfile" start
    sleep 3
else
    echo "PostgreSQL already running"
fi

# Check if database needs migration (first time setup)
if ! psql -d letta -c "SELECT 1 FROM organizations LIMIT 1;" > /dev/null 2>&1; then
    echo ""
    echo "Database tables not found. Running migrations..."
    cd "$LETTA_REPO"
    alembic upgrade head
    cd "$PROJECT_DIR"
    echo "Migrations complete!"
fi

echo ""
echo "Letta version: $(python -c 'import letta; print(letta.__version__)')"
echo ""
echo "Starting Letta server on http://127.0.0.1:8283"
echo "Press Ctrl+C to stop"
echo ""

# Start Letta server
letta server --host 127.0.0.1 --port 8283
