#!/bin/sh
# Train all model variants in parallel, one Julia process per variant.
#
# Usage (from any directory):
#   sh src/train_parallel.sh [threads per process]
#
# Default number of threads per process: number of physical CPU cores / number of variants.
# Logs: reports/report_<variant>.txt; merged and cleaned log: reports/report.txt.

set -u
cd "$(dirname "$0")/.." || exit 1

VARIANTS="threedoses totaldose totaldose_mm totaldose_morn_midday"
NV=$(echo $VARIANTS | wc -w)
CORES=$(lscpu -p=core 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "$CORES" -gt 0 ] || CORES=$(nproc)
T=${1:-$(( CORES / NV > 0 ? CORES / NV : 1 ))}

mkdir -p reports
echo "Training $NV variants in parallel, $T threads each"

PIDS=""
for v in $VARIANTS; do
    julia --startup-file=no -t "$T" src/train.jl "$v" > "reports/report_$v.txt" 2>&1 &
    PIDS="$PIDS $!"
    echo "  $v: PID $!, log: reports/report_$v.txt"
done

FAILED=""
set -- $PIDS
for v in $VARIANTS; do
    wait "$1" || FAILED="$FAILED $v"
    shift
done
if [ -n "$FAILED" ]; then
    echo "Training failed:$FAILED (see reports/report_<variant>.txt)"
    exit 1
fi

# merged log: common header (packages, polynomial models, data) from the first log, then
# the variant sections; progress bars are removed
clean() {
    awk '{ sub(/\r$/, ""); n = split($0, a, "\r"); s = a[n]
           if (s ~ /^(Optimising neural net|Evaluating over [0-9]+ metamodels):/) next
           sub(/[ \t]+$/, "", s); print s }' "$1"
}
{
    FIRST=1
    for v in $VARIANTS; do
        if [ "$FIRST" = 1 ]; then
            clean "reports/report_$v.txt" | sed -n '/^Model variant:/q;p'
            FIRST=0
        fi
        echo
        clean "reports/report_$v.txt" | sed -n '/^Model variant:/,/^Saving: cyp_levels.jld/p' | sed '$d'
    done
    echo
    echo "Training completed"
} | cat -s > reports/report.txt

echo "Done: reports/report.txt, reports/performance.csv"
