# Letta Server Setup Guide

## Overview

The Letta server (v0.16.1) runs in an **isolated conda environment** to avoid dependency conflicts with the main project's ML stack (torch, transformers, etc.).

- **Server environment**: `./letta_server_env/` (conda)
- **Client in main project**: `letta-client` package via Poetry

## Quick Start

### Method 1: Using the Helper Script (Recommended)

```bash
# Start server on default port (8283)
./start_letta_server.sh

# Start on custom port
./start_letta_server.sh --port 8284

# View help
./start_letta_server.sh --help
```

### Method 2: Manual Activation

```bash
# Load anaconda
module load anaconda/3

# Activate Letta server environment
conda activate ./letta_server_env

# Verify version
python -c "import letta; print(f'Letta version: {letta.__version__}')"

# Start server
letta server --host 127.0.0.1 --port 8283

# Stop with Ctrl+C
```

## Main Project Usage

Your main project only needs the `letta-client` package to connect:

```python
from letta_client import Letta

# Connect to local server
client = Letta(base_url="http://localhost:8283", timeout=1000)

# Create agent
agent = client.agents.create(
    name="my_agent",
    agent_type="letta_v1_agent",
    tool_rules=[],  # Empty = discretionary behavior ✅
    # ... rest of config
)
```

## Environment Details

### Letta Server Environment
- **Location**: `./letta_server_env/`
- **Type**: Conda environment
- **Python**: 3.11
- **Letta version**: 0.16.1
- **Purpose**: Run Letta server only

### Main Project Environment  
- **Location**: `./.venv/`
- **Type**: Poetry + Conda hybrid
- **Letta package**: Removed (was causing dependency conflicts)
- **Letta-client version**: >=1.6.0
- **Purpose**: Your experiments and agent code

## Benefits of This Setup

1. ✅ **No dependency conflicts**: Server and ML stack are isolated
2. ✅ **Latest Letta**: Server runs v0.16.1 with all bug fixes
3. ✅ **Discretionary behavior**: v0.16.1 properly supports `tool_rules=[]`
4. ✅ **Independent upgrades**: Can upgrade server without touching main project
5. ✅ **Clean separation**: Server is infrastructure, client is your code

## Troubleshooting

### Server won't start
```bash
# Check if port is already in use
netstat -tuln | grep 8283

# Kill existing process
pkill -f "letta server"

# Try different port
./start_letta_server.sh --port 8284
```

### Connection errors from client
```python
# Check server is running
import requests
response = requests.get("http://localhost:8283/health")
print(response.json())
```

### Version mismatch warnings
```bash
# Upgrade letta-client in main project
poetry update letta-client
```

## Server Data Location

Letta stores data in: `~/.letta/`

To backup:
```bash
cp -r ~/.letta ~/.letta.backup.$(date +%Y%m%d)
```

To reset:
```bash
rm -rf ~/.letta
# Restart server to recreate
```

## Next Steps

1. Start the server: `./start_letta_server.sh`
2. Update `letta_agent.py` to add `tool_rules=[]` parameter
3. Run your experiments!

## Key Finding from Diagnostics

**Root cause**: Letta v0.12.1 hardcoded `continue_loop` rules for `letta_v1_agent`, ignoring the `tool_rules` parameter.

**Solution**: Letta v0.16.1 properly respects `tool_rules=[]`, enabling true discretionary behavior that matches the Cloud API.

