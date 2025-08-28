#!/bin/bash

# Ensure log directories exist and are writable
mkdir -p /app/logs /app/data /app/webhook_outputs
# Create Crawl4AI data directory with proper permissions
mkdir -p /home/crawler/.crawl4ai
chown -R crawler:crawler /app/logs /app/data /app/webhook_outputs /home/crawler

# Start supervisord
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
