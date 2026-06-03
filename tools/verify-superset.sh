#!/usr/bin/env bash
# verify-superset.sh <folder> <hf_repo_id>
#
# Refuse (exit 1) if any file currently on the Hub repo is MISSING from the local
# folder, because hub-sync runs `hf upload --delete="*"` and would DELETE it on sync.
#
# Fails CLOSED on every error mode; passes for a not-yet-created (brand-new) repo.
# Runs both locally (after `git add <folder>`) and as a CI pre-step in the sync workflow.
set -euo pipefail

folder="${1:?usage: verify-superset.sh <folder> <hf_repo_id>}"
repo="${2:?usage: verify-superset.sh <folder> <hf_repo_id>}"
tok="${HF_TOKEN:-$(hf auth token 2>/dev/null)}"

code=$(curl -s -o /tmp/tree.json -w '%{http_code}' \
  -H "Authorization: Bearer $tok" \
  "https://huggingface.co/api/datasets/$repo/tree/main?recursive=true")

case "$code" in
  404) echo "✅ OK — $repo doesn't exist yet (brand-new repo); nothing on the Hub to preserve."; exit 0 ;;
  200) : ;;
  *)   echo "❌ BLOCK — Hub API returned HTTP $code for $repo; cannot verify, failing closed."; exit 1 ;;
esac

# .git*/.gitattributes are excluded by the action anyway, so don't treat them as "missing".
hub=$(python3 -c "import json;[print(f['path']) for f in json.load(open('/tmp/tree.json')) if f.get('type')=='file']" \
      | grep -vE '^\.git' | sort || true)

if [ -z "$hub" ]; then
  echo "✅ OK — $repo has no content files to preserve (empty or .gitattributes-only)."; exit 0
fi

# Tracked files == checked-out tree (CI) or staged tree (local, after `git add`).
local=$(git ls-files "$folder" | sed "s#^$folder/##" | sort)

missing=$(comm -23 <(printf '%s\n' "$hub") <(printf '%s\n' "$local"))
if [ -n "$missing" ]; then
  echo "❌ BLOCK — these Hub files are missing locally and WOULD BE DELETED on sync:"
  echo "$missing"
  exit 1
fi

echo "✅ OK — '$folder' is a superset of $repo ($(printf '%s\n' "$hub" | grep -c .) Hub files all present)."
