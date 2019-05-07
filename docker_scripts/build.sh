#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")"; cd ..; pwd)"
echo "Building starfcpy Docker image..."
docker build -t starfcpy .
