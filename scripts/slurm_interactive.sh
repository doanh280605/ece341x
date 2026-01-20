#!/usr/bin/env bash
# Request an interactive GPU session.
# Edit the partition/account/qos/time/gres to match your campus cluster.

set -euo pipefail

PARTITION="${PARTITION:-academic}"
TIME="${TIME:-02:50:00}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-4}"
MEM="${MEM:-16G}"
ACCOUNT="${ACCOUNT:-ece341x}"

echo "Requesting interactive session..."
echo "  partition: $PARTITION"
echo "  time:      $TIME"
echo "  gpus:      $GPUS"
echo "  cpus:      $CPUS"
echo "  mem:       $MEM"

srun --partition="$PARTITION" \
     --time="$TIME" \
     --gres="gpu:$GPUS" \
     --cpus-per-task="$CPUS" \
     --mem="$MEM" \
     --account="$ACCOUNT"\
     --pty bash
