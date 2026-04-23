#!/usr/bin/env bash
# Raises iogpu.wired_limit_mb to N% of physical RAM. Default 87.5%.
# Reverts automatically at reboot -- never persists.
set -euo pipefail
pct="${1:-87.5}"

# Input validation -- prevent command injection via python3 -c
if ! [[ "$pct" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "ERROR: percentage must be a number (e.g. 87.5), got: '$pct'" >&2
  exit 1
fi
if (( $(echo "$pct > 100" | bc -l) )) || (( $(echo "$pct <= 0" | bc -l) )); then
  echo "ERROR: percentage must be in (0, 100], got: $pct" >&2
  exit 1
fi

ram_mb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 ))
target=$(python3 -c "import sys; print(int(${ram_mb} * float(sys.argv[1]) / 100))" "$pct")
echo "Setting iogpu.wired_limit_mb=$target (of $ram_mb MB, ${pct}%)"
sudo sysctl "iogpu.wired_limit_mb=$target"
echo "Reverts on reboot. Current: $(sysctl -n iogpu.wired_limit_mb) MB"
