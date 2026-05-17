#!/bin/bash
# Auto-commit and push all changes with a timestamped message
git add .
git commit -m "Auto-commit: $(date '+%Y-%m-%d %H:%M:%S')"
git push
