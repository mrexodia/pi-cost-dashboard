# Pi Agent Cost Tracker

Monitor and analyze your [Pi](https://shittycodingagent.ai) coding agent API costs with detailed statistics and interactive dashboards.

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

## Installation

### Requirements

- **Python 3.12+** (for dashboard and stats tools)
- **Pi** - The coding agent ([install guide](https://github.com/mariozechner/pi-coding-agent))

### Setup

```bash
# Clone or download this repository
git clone <your-repo-url>
cd pi-cost-tracker
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

## Credits

- **[Mario Zechner](https://github.com/mariozechner)** - For Pi and its export feature
