#!/bin/bash

cp hooks/pre-push .git/hooks/pre-push

chmod +x .git/hooks/pre-push

echo "Pre-push installed."
