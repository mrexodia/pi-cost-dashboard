# Pi Agent Cost Tracker

Monitor and analyze your [Pi](https://github.com/mariozechner/pi-coding-agent) coding agent API costs with detailed statistics and interactive dashboards.

## Features

### 📊 Cost Dashboard (`cost_dashboard.py`)

**Interactive web dashboard** showing all your Pi sessions with real-time cost analysis:

```bash
./cost_dashboard.py
# Opens at http://localhost:8080
```

**Dashboard displays:**
- **Global stats** - Total spending, projects, sessions, API calls, tokens used
- **Daily spending** - Visual chart of costs over time
- **Model breakdown** - Cost per AI model (Claude Opus, Sonnet, etc.)
- **Project view** - All projects with expandable model details
- **Session browser** - Every session with resume commands and full transcripts
- **Sortable tables** - Click headers to sort by cost, tokens, time, date

**Key metrics tracked:**
- **Total cost** - Broken down by input, output, cache operations
- **Token usage** - Input, output, cache read/write tokens
- **LLM time** - Actual time the AI was working (vs. waiting)
- **Session duration** - Wall-clock time per session
- **Cache efficiency** - See how much you're saving with prompt caching

### 📈 Cost Statistics (`cost_stats.py`)

**Command-line tool** for quick cost analysis of sessions in current directory:

```bash
./cost_stats.py
```

**Sample output:**
```
============================================================
COST STATISTICS SUMMARY
============================================================

📁 Files analyzed: 11
💬 Total API calls: 234

⏱️  TIME PERIOD:
   First activity:  2025-12-24 00:47:17 UTC
   Last activity:   2025-12-26 00:03:53 UTC
   Total span:      1d 23h
   Avg cost/hour:   $0.85

📊 TOKEN USAGE:
   Input tokens:          1,234,567
   Output tokens:           234,567
   Cache read tokens:     2,345,678
   Cache write tokens:      123,456
   Total tokens:          3,938,268

💰 COSTS:
   Input cost:         $   0.1234
   Output cost:        $   5.8910
   Cache read cost:    $   1.1729
   Cache write cost:   $   0.7716
   ─────────────────────────────
   TOTAL COST:         $   7.9589

🤖 BY MODEL:
   claude-opus-4-5:
      Messages: 180  |  Tokens: 3,200,000  |  Cost: $6.5432
   claude-sonnet-4:
      Messages: 54   |  Tokens: 738,268    |  Cost: $1.4157

📅 BY DAY:
   2025-12-24:  $ 2.1234  ( 45 msgs,  1,234,567 tokens)
   2025-12-25:  $ 4.8355  (145 msgs,  2,145,678 tokens)
   2025-12-26:  $ 1.0000  ( 44 msgs,    558,023 tokens)

🕐 BY HOUR (UTC):
   14:00  $  0.52  ████████
   15:00  $  1.23  ███████████████████
   16:00  $  0.87  █████████████

📄 BY SESSION:
   12-24 00:47    1h23m  $  2.0541  ( 42 msgs)
   12-25 01:56    2h8m   $  0.8934  ( 28 msgs)
   12-25 11:04    3h26m  $  0.2412  ( 12 msgs)
   ...
============================================================
```

### 🌐 Gist Publisher (`pi-gist`)

**Export and share** Pi session transcripts as beautiful HTML on GitHub Gist:

```bash
# Export and publish latest session
./pi-gist $(ls -t *.jsonl | head -1) --gist --open

# Interactive picker
./pi-gist --interactive --gist
```

Gets you a shareable URL like: `https://gistpreview.github.io/?abc123.../session.html`

**See [Gist Publisher Documentation](#gist-publisher) below for full details.**

## Installation

### Requirements

- **Python 3.7+** (for dashboard and stats tools)
- **Pi** - The coding agent ([install guide](https://github.com/mariozechner/pi-coding-agent))
- **GitHub CLI** (optional, for gist publishing):
  ```bash
  brew install gh
  gh auth login
  ```

### Setup

```bash
# Clone or download this repository
git clone <your-repo-url>
cd pi-cost-tracker

# Make scripts executable
chmod +x cost_stats.py cost_dashboard.py pi-gist
```

## Quick Start

### Analyze Current Directory

```bash
# View stats for all .jsonl files in current directory
./cost_stats.py
```

### Launch Dashboard

```bash
# Start web dashboard (analyzes all sessions in ~/.pi/agent/sessions)
./cost_dashboard.py

# Open http://localhost:8080 in your browser

# Use custom port
./cost_dashboard.py --port 3000
```

### Share a Session

```bash
# Export most recent session to GitHub Gist
./pi-gist $(ls -t *.jsonl | head -1) --gist --open
```

## What Costs Are Tracked

Pi automatically tracks detailed API costs in every session (`.jsonl` files):

### Per-Message Costs
- **Input tokens** - Your prompts and context
- **Output tokens** - AI-generated responses
- **Cache read** - Reusing previously cached context (10% of input cost)
- **Cache write** - Building cache for future reuse (25% more than input)

### Cumulative Stats
- **Total tokens** - All token operations combined
- **Total cost** - Sum of all API charges
- **Model usage** - Breakdown by AI model (Claude Opus, Sonnet, etc.)
- **Time metrics** - Session duration and LLM working time

### Cost Optimization Tips

The dashboard helps you:
- **Monitor spending** - See which projects cost the most
- **Optimize prompts** - Identify expensive sessions
- **Track cache hits** - Maximize cache efficiency (saves 90% on repeated context)
- **Compare models** - See cost differences between Claude Opus vs Sonnet
- **Analyze patterns** - When do you code most? What does it cost?

## Gist Publisher

The `pi-gist` script exports Pi sessions to HTML and publishes them to GitHub Gist for easy sharing.

### Usage

```bash
# Export and publish to gist
./pi-gist session.jsonl --gist

# Export, publish, and open in browser
./pi-gist session.jsonl --gist --open

# Interactive picker
./pi-gist --interactive --gist --open

# Just export locally (no gist)
./pi-gist session.jsonl -o output.html

# Export latest session
./pi-gist $(ls -t *.jsonl | head -1) --gist --open

# Batch export all sessions
for f in *.jsonl; do ./pi-gist "$f" --gist; done
```

### Installation

```bash
# Install globally
./pi-gist --install
# Now use: pi-gist from anywhere

# Or setup alias
./pi-gist --setup-alias
source ~/.zshrc  # or ~/.bashrc
```

### What Gets Exported

- Session metadata (timestamp, directory, costs)
- Full message history with syntax highlighting
- Tool executions (bash, read, write, edit)
- Thinking blocks (if present)
- Expandable/collapsible outputs
- Dark theme optimized for readability
- **Cost breakdown** - Detailed token usage and spending

## Data Structure

Pi sessions are stored in `~/.pi/agent/sessions/<project>/<session>.jsonl`

Each line is a JSON object representing a message or event. Cost data is embedded in messages with `usage` and `cost` fields:

```json
{
  "type": "message",
  "timestamp": "2025-12-26T15:30:00.000Z",
  "message": {
    "role": "assistant",
    "model": "claude-opus-4-5",
    "usage": {
      "input": 1234,
      "output": 567,
      "cacheRead": 5000,
      "cacheWrite": 0,
      "totalTokens": 6801
    },
    "cost": {
      "input": 0.0185,
      "output": 0.0142,
      "cacheRead": 0.0025,
      "cacheWrite": 0.0000,
      "total": 0.0352
    }
  }
}
```

## Examples

See example session exports:
- https://gist.github.com/mrexodia/a17e6e07d23209b5cc83deb34768a298
- Preview: https://gistpreview.github.io/?a17e6e07d23209b5cc83deb34768a298/test-pi-export.html

## Troubleshooting

**"No JSONL session files found"**
- Run `cost_stats.py` from a directory containing `.jsonl` files
- Or use `cost_dashboard.py` which reads from `~/.pi/agent/sessions/`

**Dashboard shows no data**
- Check that `~/.pi/agent/sessions/` exists and contains session folders
- Make sure you've run some Pi sessions that generated cost data

**"gh: command not found"** (for gist publishing)
```bash
brew install gh
gh auth login
```

## Contributing

Contributions welcome! This project focuses on:
- Accurate cost tracking and analysis
- Clear data visualization
- Simple, maintainable code
- Cross-platform compatibility

## Credits

- **[Mario Zechner](https://github.com/mariozechner)** - For Pi and its export feature
- **[Simon Willison](https://simonwillison.net/)** - For claude-code-transcripts inspiration
- **[GitHub](https://github.com)** - For Gist and the gh CLI

## License

Apache 2.0
