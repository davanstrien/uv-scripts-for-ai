# CLAUDE.md

The operational runbook for this repo lives in **[AGENTS.md](AGENTS.md)** — read it before changing anything.

⛔ One rule above all: **never delete a file from the Hub.** The `hub-sync` Action mirrors deletions, so a
repo gets a sync workflow only after its folder is a verified *superset* of that repo's current Hub contents
(seed-then-flip). Seed with `hf download`, drop the downloaded `.gitattributes`, curate additively, and run
`tools/verify-superset.sh` before flipping.
