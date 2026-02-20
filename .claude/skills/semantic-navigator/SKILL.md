# Semantic Navigator

Label a repository's files semantically for the semantic-navigator TUI.

## Arguments

$ARGUMENTS = repository path (required), plus optional flags like --gpu

## Process

### Step 1: Generate clusters

Run the embedding and clustering pipeline (no LLM needed):

```bash
 uv run --project "C:\Projects\semantic-navigator" semantic-navigator $ARGUMENTS --export-clusters clusters.json
```

### Step 2: Extract file paths

`clusters.json` may be too large to read directly (it embeds file content). Extract just the structure you need:

```python
import json
with open('clusters.json') as f:
    data = json.load(f)
# Print leaf clusters with file paths only
for i, c in enumerate(data['leaf_clusters']):
    files = [f['path'] for f in c['files']]
    print(f'CLUSTER {i} [{c["cluster_id"]}]:')
    for fp in files:
        print(f'  {fp}')
# Print hierarchy
for h in data['hierarchy']:
    children = [child['cluster_id'] for child in h['children']]
    print(f'HIERARCHY {h["cluster_id"]}: {children}')
```

### Step 3: Label leaf clusters

For each leaf cluster, label every file with:

- **overarchingTheme**: broad domain (use CONSISTENT terms — pick ONE term and reuse it, don't alternate synonyms like "Styling"/"Web Design"/"UI Styling")
- **distinguishingFeature**: what makes THIS file unique vs its siblings (distinguish by role: impl vs interface vs adapter vs test vs config)
- **label**: 3-7 word purpose (be specific: "Rate Limiting Middleware" not "Server Task")

Rules:
- Use filenames for context but don't copy paths verbatim into labels
- Config/dotfiles: theme = their role ("Build Configuration", "Editor Settings"), NOT their content
- Test files: must be distinguished from the code they test
- Re-export/barrel files: label as "Module Barrel Export" or similar
- Each label must be unique across all files

Label formatting rules:
- Split PascalCase/camelCase into separate words: `ServerCreationModal` → "Server Creation Modal"
- Don't prefix labels with directory names that add no meaning: "Bus Enhanced Event Bus" → "Enhanced Event Bus"
- Don't repeat the same word from the module name and the filename: "Economy Routes Routes" → "Economy Routes"
- Every label must be at least 3 words. No single-word labels like "App" or "Server"
- No label may have theme = "Miscellaneous" or "Other" — always find a meaningful theme

Anti-stutter rules:
- When prepending a module or directory name to a label, check if any word from that prefix already appears anywhere in the rest of the label. If so, drop the prefix or the duplicate word.
- Check for near-duplicates from word stems, not just exact matches: "Investigate" and "Investigation" count as stutter. "Handler Handler" counts. "Bank Crack Bank Account" has "Bank" twice.
- After generating all labels, run a programmatic stutter check (see Step 6).

Label meaning rules:
- Labels must describe **what the file does**, not just echo the filename. Bad: "Core Shared Common", "Core Shared Brand". Good: "Shared Enum Definitions", "Branded Type Factory".
- When a filename is generic (common.ts, types.ts, utils.ts, helpers.ts), **read the file** (or use its cluster context) to assign a meaningful label. Echoing the filename is not acceptable for generic names.
- Assign themes based on what the code **actually does**, not where it lives. A file under `/logic/` that only runs SQL queries is "Database Layer", not "Game Logic". A file under `/repository/` that contains domain validation is "Domain Model", not "Data Access".

Theme granularity:
- Aim for 10-30 distinct themes across the whole repo
- If any single theme covers more than ~15% of files, split it into more specific sub-themes (e.g. "Core Infrastructure" → "Event System", "HTTP Framework", "Caching Layer")

### Step 4: Label hierarchy clusters

For each entry in the `hierarchy` array, label each child based on the child_labels (the file labels you already assigned), NOT file paths. Use:

- **overarchingTheme**: broad domain shared by child clusters
- **distinguishingFeature**: what distinguishes THIS cluster from its siblings
- **label**: 2-4 word category name

Hierarchy labels must describe the **semantic role** of each child cluster, not repeat the most-common theme with a number. Each sibling label should answer "what would I find here that I wouldn't find in the other siblings?"

- Bad: "Unit Testing (2)", "Database Migration (3)", "Game Logic (1)"
- Good: "Hacking & Access Control Tests", "Bank Schema Migrations", "NPC Decision Engine"

### Step 5: Write labels

Write the labels to `labels.json` in this exact format:

```json
{
  "model_identity": "<copy from clusters.json>",
  "file_labels": {
    "src/main.py": {"overarchingTheme": "...", "distinguishingFeature": "...", "label": "..."},
    "src/util.py": {"overarchingTheme": "...", "distinguishingFeature": "...", "label": "..."}
  },
  "cluster_labels": {
    "<cluster_id from hierarchy>": [
      {"overarchingTheme": "...", "distinguishingFeature": "...", "label": "..."}
    ]
  }
}
```

For `cluster_labels`: each key is the `cluster_id` of a hierarchy entry, and the value is an array with one label per child (in the same order as the `children` array).

### Step 6: Validate labels

Before importing, validate your labels programmatically AND by spot-checking.

**Programmatic checks (mandatory):**
1. **Stutter check**: For each label, check if any word (case-insensitive) appears more than once. Also check stems: if removing common suffixes (-tion, -ing, -ment, -er, -or, -ive, -ly) from two words produces the same root, that's stutter.
2. **Short label check**: Every label must be at least 3 words.
3. **Theme distribution**: No theme may exceed ~15% of total files.
4. **No "Miscellaneous"**: Every file must have a meaningful theme.
5. **Uniqueness**: All labels must be unique.

**Spot-check (mandatory):**
1. **Read 5-10 actual files** from diverse clusters and verify the label accurately describes what the code does (not just what the filename says). Pay special attention to generically-named files (common.ts, types.ts, utils.ts, helpers.ts, index.ts).
2. **Verify all hierarchy labels** — none should be numbered themes.
3. **Check theme accuracy for edge cases**: files in /logic/ that do SQL, files in /repository/ that contain domain logic, admin-only files labeled as "Game Logic", etc.

Fix any issues before proceeding.

### Step 7: Import and display

```bash
uv run semantic-navigator $ARGUMENTS --import-labels labels.json
```

### Approach: script vs. direct

- **Under ~100 files**: write `labels.json` directly using the Write tool, in batches if needed
- **Over ~100 files**: you may write a labeling script, but you **must** run the Step 6 validation afterward and fix any patterns the script produced poorly. Common script pitfalls:
  - Naive title-casing of camelCase
  - Directory-name prefixes leaking into labels ("Bus Enhanced Event Bus")
  - Module name + filename stutter ("Economy Economy Service")
  - Double suffixes ("Handler Handler", "Tests Tests")
  - Catch-all fallback themes ("Miscellaneous", "Core Infrastructure")
  - Echoing generic filenames verbatim ("Core Shared Common" instead of describing what common.ts contains)
