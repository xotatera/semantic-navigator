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

### Step 2: Label leaf clusters

Read `clusters.json`. For each leaf cluster, label every file with:

- **overarchingTheme**: broad domain (use CONSISTENT terms — pick ONE term and reuse it, don't alternate synonyms like "Styling"/"Web Design"/"UI Styling")
- **distinguishingFeature**: what makes THIS file unique vs its siblings (distinguish by role: impl vs interface vs adapter vs test vs config)
- **label**: 3-7 word purpose (be specific: "Rate Limiting Middleware" not "Server Task")

Rules:
- Use filenames for context but don't copy paths verbatim into labels
- Config/dotfiles: theme = their role ("Build Configuration", "Editor Settings"), NOT their content
- Test files: must be distinguished from the code they test
- Re-export/barrel files: label as "Module Barrel Export" or similar
- Each label must be unique across all files

### Step 3: Label hierarchy clusters

For each entry in the `hierarchy` array, label each child based on the child_labels (the file labels you already assigned), NOT file paths. Use:

- **overarchingTheme**: broad domain shared by child clusters
- **distinguishingFeature**: what distinguishes THIS cluster from its siblings
- **label**: 2-4 word category name

### Step 4: Write labels

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

### Step 5: Import and display

```bash
uv run semantic-navigator $ARGUMENTS --import-labels labels.json
```
