#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./run-demos.sh [-n count] [-p python_command]

Runs demo scripts v1 through v6, excluding v2-tracing.

Options:
  -n count           Number of times to run the full demo sequence (default: 1)
  -p python_command  Python executable to use (default: python)
  -h                 Show this help message

Examples:
  ./run-demos.sh -n 3
  ./run-demos.sh -n 2 -p python3
USAGE
}

run_count=1
python_cmd="python"

while getopts ":n:p:h" opt; do
  case "$opt" in
    n)
      run_count="$OPTARG"
      ;;
    p)
      python_cmd="$OPTARG"
      ;;
    h)
      usage
      exit 0
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage >&2
      exit 2
      ;;
    \?)
      echo "Unknown option: -$OPTARG" >&2
      usage >&2
      exit 2
      ;;
  esac
done

shift $((OPTIND - 1))

if [ "$#" -ne 0 ]; then
  echo "Unexpected argument: $1" >&2
  usage >&2
  exit 2
fi

case "$run_count" in
  ''|*[!0-9]*)
    echo "-n must be a positive integer." >&2
    exit 2
    ;;
esac

if [ "$run_count" -lt 1 ]; then
  echo "-n must be at least 1." >&2
  exit 2
fi

scripts=(
  "openai-agent-demo-v1.py"
  "openai-agent-demo-v2.py"
  "openai-agent-demo-v3.py"
  "openai-agent-demo-v4.py"
  "openai-agent-demo-v5.py"
  "openai-agent-demo-v6.py"
)

for ((run = 1; run <= run_count; run++)); do
  printf '\n=== Demo run %d/%d ===\n' "$run" "$run_count"

  for script in "${scripts[@]}"; do
    printf '\n--- %s ---\n' "$script"
    "$python_cmd" "$script"
  done
done
