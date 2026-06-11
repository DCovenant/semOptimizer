#!/bin/bash
# Log system/GPU/cgroup memory every 2s so the next OOM is attributable.
# Usage: ./tools/memwatch.sh [logfile]   (default /tmp/memwatch.log — survives
# a session crash; the box doesn't reboot, only the desktop dies)
LOG=${1:-/tmp/memwatch.log}
GPU=$(ls -d /sys/class/drm/card*/device | head -1)
echo "=== memwatch start $(date -Is) ===" >> "$LOG"
while true; do
    {
    echo "--- $(date -Is)"
    awk '/MemAvailable|SwapFree|^Shmem:|^Mapped/ {print}' /proc/meminfo
    echo "GTT_used_MB $(( $(cat $GPU/mem_info_gtt_used) / 1048576 ))  VRAM_used_MB $(( $(cat $GPU/mem_info_vram_used) / 1048576 ))"
    # per-cgroup memory for the user session (includes the carla scope)
    for cg in /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/*/; do
        cur=$(cat "$cg/memory.current" 2>/dev/null) || continue
        swp=$(cat "$cg/memory.swap.current" 2>/dev/null || echo 0)
        [ "$cur" -gt 209715200 ] && echo "cg $(basename $cg) mem_MB $((cur/1048576)) swap_MB $((swp/1048576))"
    done
    ps -eo rss=,comm= --sort=-rss | head -6 | awk '{printf "proc %s rss_MB %d\n", $2, $1/1024}'
    } >> "$LOG"
    sleep 2
done
