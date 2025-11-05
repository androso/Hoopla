# Hoopla - Python Project

## Overview
Basic Python project named "hoopla" using the uv package manager. The project prints "hello world" to the console.

## Project Structure
- `main.py` - Main entry point that prints "hello world"
- `pyproject.toml` - Project configuration for uv package manager
- `.gitignore` - Standard Python gitignore file
- `.python-version` - Python version specification

## Technology Stack
- Python 3.12
- uv package manager (v0.9.5)
- Go 1.24.4 (for bootdev CLI support)
- No external Python dependencies

## Running the Project
The project is configured with a workflow that runs:
```bash
uv run main.py
```

## Project Configuration
Project name: hoopla
Python version: >= 3.12
Dependencies: None (minimal setup)

## Recent Changes
- 2025-11-05: Removed duplicate bootdev installation - now using only Go-installed v1.20.5 from ~/go/bin
- 2025-11-05: Updated .profile to include ~/go/bin in PATH for go install commands
- 2025-11-01: Initial project setup with uv package manager
- Created basic main.py file that prints "hello world"
- Configured workflow to run the application
- Installed Go 1.24.4 to support bootdev CLI commands
