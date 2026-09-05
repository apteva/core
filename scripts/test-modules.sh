#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
while IFS= read -r mod; do
  [[ "$mod" == "go.mod" ]] && continue
  (cd "$(dirname "$mod")" && env GOWORK=off go test -short -count=1 -timeout=120s ./...)
done < <(rg --files -g go.mod | sort)
