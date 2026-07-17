#!/bin/bash
set -e

# Execute the parent monitoring script which subsequently spawns Rasa
exec python core_monitor.py