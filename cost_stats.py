#!/usr/bin/env python3
"""Analyze JSONL files and print cost statistics."""

import json
import glob
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

def parse_timestamp(ts):
    """Parse ISO timestamp string to datetime."""
    if not ts:
        return None
    try:
        # Handle formats like "2025-12-25T23:03:53.010Z"
        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except:
        return None

def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def analyze_jsonl_files():
    files = glob.glob("*.jsonl")
    
    if not files:
        print("No JSONL files found in current directory")
        return
    
    total_stats = {
        "files": 0,
        "messages": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_cache_read": 0.0,
        "cost_cache_write": 0.0,
        "cost_total": 0.0,
    }
    
    model_stats = defaultdict(lambda: {"messages": 0, "tokens": 0, "cost": 0.0})
    daily_stats = defaultdict(lambda: {"messages": 0, "tokens": 0, "cost": 0.0})
    hourly_stats = defaultdict(lambda: {"messages": 0, "tokens": 0, "cost": 0.0})
    file_stats = []
    
    all_timestamps = []
    
    for filepath in sorted(files):
        file_cost = 0.0
        file_tokens = 0
        file_messages = 0
        file_start = None
        file_end = None
        
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("type") == "message" and "message" in data:
                        msg = data["message"]
                        if "usage" in msg:
                            usage = msg["usage"]
                            cost = usage.get("cost", {})
                            model = msg.get("model", "unknown")
                            
                            total_stats["messages"] += 1
                            total_stats["input_tokens"] += usage.get("input", 0)
                            total_stats["output_tokens"] += usage.get("output", 0)
                            total_stats["cache_read_tokens"] += usage.get("cacheRead", 0)
                            total_stats["cache_write_tokens"] += usage.get("cacheWrite", 0)
                            total_stats["total_tokens"] += usage.get("totalTokens", 0)
                            total_stats["cost_input"] += cost.get("input", 0)
                            total_stats["cost_output"] += cost.get("output", 0)
                            total_stats["cost_cache_read"] += cost.get("cacheRead", 0)
                            total_stats["cost_cache_write"] += cost.get("cacheWrite", 0)
                            total_stats["cost_total"] += cost.get("total", 0)
                            
                            file_cost += cost.get("total", 0)
                            file_tokens += usage.get("totalTokens", 0)
                            file_messages += 1
                            
                            model_stats[model]["messages"] += 1
                            model_stats[model]["tokens"] += usage.get("totalTokens", 0)
                            model_stats[model]["cost"] += cost.get("total", 0)
                            
                            # Track timestamps
                            ts = parse_timestamp(data.get("timestamp"))
                            if ts:
                                all_timestamps.append(ts)
                                if file_start is None or ts < file_start:
                                    file_start = ts
                                if file_end is None or ts > file_end:
                                    file_end = ts
                                
                                # Daily stats
                                day_key = ts.strftime("%Y-%m-%d")
                                daily_stats[day_key]["messages"] += 1
                                daily_stats[day_key]["tokens"] += usage.get("totalTokens", 0)
                                daily_stats[day_key]["cost"] += cost.get("total", 0)
                                
                                # Hourly stats
                                hour_key = ts.hour
                                hourly_stats[hour_key]["messages"] += 1
                                hourly_stats[hour_key]["tokens"] += usage.get("totalTokens", 0)
                                hourly_stats[hour_key]["cost"] += cost.get("total", 0)
                            
                except json.JSONDecodeError:
                    continue
        
        total_stats["files"] += 1
        duration = (file_end - file_start).total_seconds() if file_start and file_end else 0
        file_stats.append({
            "file": Path(filepath).name,
            "messages": file_messages,
            "tokens": file_tokens,
            "cost": file_cost,
            "start": file_start,
            "end": file_end,
            "duration": duration
        })
    
    # Print results
    print("=" * 60)
    print("COST STATISTICS SUMMARY")
    print("=" * 60)
    
    print(f"\n📁 Files analyzed: {total_stats['files']}")
    print(f"💬 Total API calls: {total_stats['messages']}")
    
    # Time period stats
    if all_timestamps:
        first_ts = min(all_timestamps)
        last_ts = max(all_timestamps)
        total_span = (last_ts - first_ts).total_seconds()
        
        print(f"\n⏱️  TIME PERIOD:")
        print(f"   First activity:  {first_ts.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"   Last activity:   {last_ts.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"   Total span:      {format_duration(total_span)}")
        if total_span > 0:
            cost_per_hour = total_stats['cost_total'] / (total_span / 3600)
            print(f"   Avg cost/hour:   ${cost_per_hour:.2f}")
    
    print(f"\n📊 TOKEN USAGE:")
    print(f"   Input tokens:       {total_stats['input_tokens']:>12,}")
    print(f"   Output tokens:      {total_stats['output_tokens']:>12,}")
    print(f"   Cache read tokens:  {total_stats['cache_read_tokens']:>12,}")
    print(f"   Cache write tokens: {total_stats['cache_write_tokens']:>12,}")
    print(f"   Total tokens:       {total_stats['total_tokens']:>12,}")
    
    print(f"\n💰 COSTS:")
    print(f"   Input cost:         ${total_stats['cost_input']:>10.4f}")
    print(f"   Output cost:        ${total_stats['cost_output']:>10.4f}")
    print(f"   Cache read cost:    ${total_stats['cost_cache_read']:>10.4f}")
    print(f"   Cache write cost:   ${total_stats['cost_cache_write']:>10.4f}")
    print(f"   ─────────────────────────────")
    print(f"   TOTAL COST:         ${total_stats['cost_total']:>10.4f}")
    
    if model_stats:
        print(f"\n🤖 BY MODEL:")
        for model, stats in sorted(model_stats.items(), key=lambda x: -x[1]["cost"]):
            print(f"   {model}:")
            print(f"      Messages: {stats['messages']:,}  |  Tokens: {stats['tokens']:,}  |  Cost: ${stats['cost']:.4f}")
    
    if daily_stats:
        print(f"\n📅 BY DAY:")
        for day, stats in sorted(daily_stats.items()):
            print(f"   {day}:  ${stats['cost']:>8.4f}  ({stats['messages']:>4} msgs, {stats['tokens']:>10,} tokens)")
    
    if hourly_stats:
        print(f"\n🕐 BY HOUR (UTC):")
        # Show as a simple bar chart
        max_cost = max(s["cost"] for s in hourly_stats.values()) if hourly_stats else 1
        for hour in range(24):
            if hour in hourly_stats:
                stats = hourly_stats[hour]
                bar_len = int(20 * stats["cost"] / max_cost) if max_cost > 0 else 0
                bar = "█" * bar_len
                print(f"   {hour:02d}:00  ${stats['cost']:>7.2f}  {bar}")
    
    print(f"\n📄 BY SESSION:")
    for fs in sorted(file_stats, key=lambda x: -x["cost"]):
        duration_str = format_duration(fs['duration']) if fs['duration'] else "n/a"
        start_str = fs['start'].strftime('%m-%d %H:%M') if fs['start'] else ""
        print(f"   {start_str}  {duration_str:>6}  ${fs['cost']:>8.4f}  ({fs['messages']:>3} msgs)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    analyze_jsonl_files()
