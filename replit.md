# Hoopla - Python Project

## Overview
Python project named "hoopla" using the uv package manager. Currently contains a keyword search CLI for movies using an inverted index.

## Project Structure
- `cli/` - Command-line interface code
  - `keyword_search_cli.py` - Main CLI for movie search with inverted index
  - `test.py` - Tests
- `data/` - Data files
  - `movies.json` - Movie database
  - `stopwords.txt` - Stop words for search
- `cache/` - Cached index files
  - `index.pkl` - Inverted index cache
  - `docmap.pkl` - Document map cache
- `scripts/` - Utility scripts
  - `bootstrap_bootdev.sh` - Bootdev CLI updater
- `pyproject.toml` - Project configuration for uv package manager
- `.gitignore` - Standard Python gitignore file

## Technology Stack
- Python 3.12
- uv package manager (v0.9.5)
- Go 1.24.4 (for bootdev CLI support)
- NLTK 3.9.1 (for text processing and stemming)

## Project Configuration
Project name: hoopla
Python version: >= 3.12
Dependencies: nltk==3.9.1

## Bootdev CLI Setup

The bootdev CLI is installed via Go and managed through a bootstrap script for persistence:

### Update bootdev to latest version:
```bash
./scripts/bootstrap_bootdev.sh
```

This script:
- Downloads the latest bootdev version via `go install`
- Creates a symlink in `~/.local/bin` (which is always on PATH)
- Ensures bootdev persists across Replit restarts

Run this script anytime you want to update bootdev to the latest version.

## Recent Changes
- 2025-11-05: Created permanent bootdev installation via bootstrap script and ~/.local/bin symlink
- 2025-11-05: Removed duplicate bootdev installation - now using only Go-installed v1.20.5 from ~/go/bin
- 2025-11-01: Initial project setup with uv package manager
- Created basic main.py file that prints "hello world"
- Configured workflow to run the application
- Installed Go 1.24.4 to support bootdev CLI commands
