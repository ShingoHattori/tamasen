#!/bin/bash
# iPad の TamasenScan から記録を USB で取り込む。使い方: scan/pull.sh [セッション名…]（省略時はすべて）
# 取り込み先: scan/captures/<セッション名>/
set -e
D=${TSCAN_DEVICE:-Casper}
APP=jp.ac.tsukuba.hpcs.tamasen.scan
OUT=$(cd "$(dirname "$0")" && pwd)/captures
mkdir -p "$OUT"
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  names=($(xcrun devicectl device info files --device "$D" --domain-type appDataContainer --domain-identifier $APP 2>/dev/null \
    | awk '{print $1}' | grep -E '^Documents/scan-[0-9-]+$' | sed 's|Documents/||'))
fi
for n in "${names[@]}"; do
  if [ -f "$OUT/$n/session.json" ] && [ -d "$OUT/$n/frames" ]; then echo "skip ${n}（取り込み済み）"; continue; fi
  xcrun devicectl device copy from --device "$D" --domain-type appDataContainer --domain-identifier $APP \
    --source "Documents/$n" --destination "$OUT/$n" >/dev/null
  echo "$n: $(ls "$OUT/$n/frames" | grep -c jpg) 枚"
done
