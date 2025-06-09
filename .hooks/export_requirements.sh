#!/bin/bash
set -e

uv export > requirements.txt
git add requirements.txt
