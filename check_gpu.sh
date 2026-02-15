#!/bin/bash
# check_gpu.sh - Check GPU availability on shared server
# Usage: ./check_gpu.sh

echo "=== GPU Availability Check ==="
echo ""

# Show all GPUs with memory and utilization
echo "All GPUs status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu --format=csv

echo ""
echo "=== Free GPUs (>20GB available) ==="
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader,nounits | \
  awk -F', ' '$3 > 20000 {printf "✅ GPU %s: %s (%.1f GB free, %s%% util)\n", $1, $2, $3/1024, $4}'

FREE_COUNT=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' '$2 > 20000' | wc -l)

echo ""
if [ "$FREE_COUNT" -eq 0 ]; then
  echo "❌ No free GPUs available (all have <20GB free memory)"
  echo "   Wait for other jobs to finish or use CPU mode"
else
  echo "✅ Found $FREE_COUNT free GPU(s)"
  echo "   You can run the pipeline now"
fi
