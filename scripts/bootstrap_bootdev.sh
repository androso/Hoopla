#!/bin/bash
set -e

if ! command -v go &> /dev/null; then
    echo "❌ Error: Go is not installed. Please install Go first."
    exit 1
fi

echo "🔄 Updating bootdev CLI to latest version..."
go install github.com/bootdotdev/bootdev@latest

if [ ! -f ~/go/bin/bootdev ]; then
    echo "❌ Error: bootdev installation failed."
    exit 1
fi

echo "🔗 Creating symlink in ~/.local/bin..."
mkdir -p ~/.local/bin
ln -sf ~/go/bin/bootdev ~/.local/bin/bootdev

echo "✅ bootdev updated successfully!"
bootdev --version
