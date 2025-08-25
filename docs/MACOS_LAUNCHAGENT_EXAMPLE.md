# macOS LaunchAgent Example (embed-rerank)

This LaunchAgent ensures the server starts at user login, keeps running (auto‑restart), and logs to `/tmp`.

## File Path
Place at: `~/Library/LaunchAgents/com.embed-rerank.server.plist`

## Contents

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
        <key>Label</key>
        <string>com.embed-rerank.server</string>
        <key>ProgramArguments</key>
        <array>
                <string>/bin/sh</string>
                <string>/Users/username/embed-rerank/tools/server-launchd.sh</string>
        </array>
        <key>WorkingDirectory</key>
        <string>/Users/username/embed-rerank</string>
        <key>EnvironmentVariables</key>
        <dict>
                <key>HOST</key><string>0.0.0.0</string>
                <key>PORT</key><string>11436</string>
                <key>MODEL_NAME</key><string>mlx-community/Qwen3-Embedding-4B-4bit-DWQ</string>
        </dict>
        <key>KeepAlive</key>
        <true/>
        <key>RunAtLoad</key>
        <true/>
        <key>StandardOutPath</key>
        <string>/tmp/embed-rerank.log</string>
        <key>StandardErrorPath</key>
        <string>/tmp/embed-rerank.err</string>
</dict>
</plist>
```

## Usage

Load (macOS 12+):
```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.embed-rerank.server.plist
launchctl kickstart -k gui/$(id -u)/com.embed-rerank.server
```

View status:
```bash
launchctl print gui/$(id -u)/com.embed-rerank.server
```

Tail logs:
```bash
tail -f /tmp/embed-rerank.log /tmp/embed-rerank.err
```

Unload / stop:
```bash
launchctl bootout gui/$(id -u)/com.embed-rerank.server
```

Reload after editing:
```bash
launchctl bootout gui/$(id -u)/com.embed-rerank.server || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.embed-rerank.server.plist
```

## Notes
- Ensure the path contains no typos (spaces are fine because we invoke via bash -lc).
- `.venv` and `.env` must exist in the repo root.
- For system-wide usage, create a LaunchDaemon under `/Library/LaunchDaemons/` (then adjust ownership/root requirements).