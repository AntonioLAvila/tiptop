#!/bin/bash

set -eo pipefail

# Verify we're running from project root (check for tiptop-specific files)
if [ ! -f "tiptop/__init__.py" ]; then
    echo "ERROR: This script must be run from the tiptop project root directory"
    echo "Please run: pixi run install-cutamp"
    exit 1
fi

FORK="AntonioLAvila/cuTAMP"
REPO_URL="https://github.com/$FORK.git"
BRANCH="rlg-franka"
INSTALL_DIR="cutamp"

echo "==> Installing cuTAMP ($FORK@$BRANCH)..."

# Check if directory exists and is a healthy git repo
should_clone=true
if [ -d "$INSTALL_DIR" ]; then
    cd "$INSTALL_DIR"
    if git fsck --full &> /dev/null; then
        if [[ "$(git remote get-url origin)" != *"$FORK"* ]]; then
            echo "Pointing origin at $REPO_URL (was $(git remote get-url origin))"
            git remote set-url origin "$REPO_URL"
        fi
        echo "Updating existing cuTAMP repository to origin/$BRANCH..."
        git fetch origin "$BRANCH"
        git checkout "$BRANCH"
        git pull --ff-only origin "$BRANCH"
        should_clone=false
        cd ..
    else
        echo "✗ cuTAMP repository is corrupted, removing..."
        cd ..
        rm -rf "$INSTALL_DIR"
    fi
fi

# Clone the fork's branch
if [ "$should_clone" = true ]; then
    echo "Cloning cuTAMP $BRANCH..."
    git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi

# Install
cd "$INSTALL_DIR"
echo "Installing cuTAMP..."
pip install -e . --no-build-isolation --no-deps
cd ..
echo "✓ cuTAMP $BRANCH installed successfully: $(git -C "$INSTALL_DIR" rev-parse --short HEAD)"
