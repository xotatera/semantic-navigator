import argparse
import asyncio
import shlex
import shutil
import time

import textual
import textual.app
import textual.widgets

from semantic_navigator.cache import content_hash, label_cache_dir, list_cached_keys, load_cached_label, repo_cache_dir
from semantic_navigator.gpu import list_devices
from semantic_navigator.inference import initialize
from semantic_navigator.models import Facets, Tree
from semantic_navigator.pipeline import _generate_paths, build_cluster_tree, embed, export_cluster_tree, import_labels, tree
from semantic_navigator.util import _fmt_time, timed


class UI(textual.app.App):
    BINDINGS = [("slash", "focus_search", "Search"), ("escape", "clear_search", "Clear")]

    def __init__(self, tree_):
        super().__init__()
        self.tree_ = tree_

    async def on_mount(self):
        self.search_input = textual.widgets.Input(placeholder="Search (press / to focus)...")
        self.search_input.display = False
        self.treeview = textual.widgets.Tree(f"{self.tree_.label} ({len(self.tree_.files)})")
        self._build_tree()
        await self.mount(self.search_input)
        await self.mount(self.treeview)

    def _build_tree(self, filter_text: str = ""):
        self.treeview.clear()
        self.treeview.root.set_label(f"{self.tree_.label} ({len(self.tree_.files)})")

        def matches(child: Tree, text: str) -> bool:
            if text in child.label.lower():
                return True
            return any(text in f.lower() for f in child.files)

        def loop(node, children):
            for child in children:
                if filter_text and not matches(child, filter_text):
                    continue
                if len(child.files) <= 1:
                    n = node.add(child.label)
                    n.allow_expand = False
                else:
                    n = node.add(f"{child.label} ({len(child.files)})")
                    n.allow_expand = True
                    loop(n, child.children)

        loop(self.treeview.root, self.tree_.children)
        if filter_text:
            self.treeview.root.expand_all()

    def action_focus_search(self):
        self.search_input.display = True
        self.search_input.focus()

    def action_clear_search(self):
        self.search_input.value = ""
        self.search_input.display = False
        self._build_tree()
        self.treeview.focus()

    def on_input_changed(self, event: textual.widgets.Input.Changed):
        self._build_tree(event.value.strip().lower())


def _handle_erase_models():
    """Handle the --erase-models command."""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        if not cache_info.repos:
            print("No downloaded models found.")
            return
        total_size = sum(r.size_on_disk for r in cache_info.repos)
        print(f"Downloaded models ({total_size / 1e9:.1f} GB):")
        for repo in sorted(cache_info.repos, key=lambda r: r.size_on_disk, reverse=True):
            print(f"  {repo.repo_id} ({repo.size_on_disk / 1e9:.1f} GB)")
        confirm = input("\nDelete all downloaded models? [y/N] ").strip().lower()
        if confirm == "y":
            delete_strategy = cache_info.delete_revisions(
                [r.commit_hash for repo in cache_info.repos for r in repo.revisions]
            )
            delete_strategy.execute()
            print("Done.")
        else:
            print("Aborted.")
    except ImportError:
        print("huggingface_hub is not installed.")


def _flush_cache(repository: str):
    """Delete cached labels for the given repository (preserves embeddings)."""
    repo_dir = repo_cache_dir(repository)
    if repo_dir.exists():
        size = sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file())
        shutil.rmtree(repo_dir)
        print(f"Flushed repo cache for {repository} ({size / 1e6:.1f} MB)")
    else:
        print(f"No cache found.")


def _show_status(repository: str):
    """Show cache status for a repository."""
    import os
    from pathlib import Path

    file_paths = _generate_paths(repository)

    # Read files and compute content hashes
    file_hashes: list[tuple[str, str]] = []  # (path, hash)
    for path in file_paths:
        try:
            absolute_path = os.path.join(repository, path)
            with open(absolute_path, "rb") as f:
                text = f.read().decode("utf-8")
            file_hashes.append((path, content_hash(f"{path}:\n\n{text}")))
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError, PermissionError):
            pass

    total = len(file_hashes)

    print(f"Repository: {repository} ({total} files)")
    print()

    # Label caches
    repo_dir = repo_cache_dir(repository)
    labels_dir = repo_dir / "labels"
    if not labels_dir.exists():
        print("Labels: no label caches found")
        return

    label_dirs = [d for d in labels_dir.iterdir() if d.is_dir()]
    if not label_dirs:
        print("Labels: no label caches found")
        return

    print(f"Labels ({len(label_dirs)} model {'identity' if len(label_dirs) == 1 else 'identities'}):")
    for ldir in sorted(label_dirs):
        cached_keys = list_cached_keys(ldir, ".json")
        lbl_cached = sum(1 for _, h in file_hashes if h in cached_keys)
        lbl_missing = total - lbl_cached
        status = f"  [{ldir.name}]: {lbl_cached}/{total} cached"
        if lbl_missing > 0:
            status += f" ({lbl_missing} missing)"
        print(status)
        if lbl_missing > 0:
            print("  Missing:")
            for path, h in file_hashes:
                if h not in cached_keys:
                    print(f"    {path}")


def _export_toml(repository: str, output_path: str):
    """Export cached labels as TOML to the given file."""
    import os
    import sys

    file_paths = _generate_paths(repository)

    # Read files and compute content hashes
    file_hashes: list[tuple[str, str]] = []
    for path in file_paths:
        try:
            absolute_path = os.path.join(repository, path)
            with open(absolute_path, "rb") as f:
                text = f.read().decode("utf-8")
            file_hashes.append((path, content_hash(f"{path}:\n\n{text}")))
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError, PermissionError):
            pass

    # Find label cache dirs
    repo_dir = repo_cache_dir(repository)
    labels_dir = repo_dir / "labels"
    if not labels_dir.exists():
        print("No label caches found.", file=sys.stderr)
        return

    label_dirs = [d for d in labels_dir.iterdir() if d.is_dir()]
    if not label_dirs:
        print("No label caches found.", file=sys.stderr)
        return

    # Pick model identity
    if len(label_dirs) == 1:
        chosen = label_dirs[0]
    else:
        print("Multiple model identities found:", file=sys.stderr)
        for i, ldir in enumerate(sorted(label_dirs)):
            cached_keys = list_cached_keys(ldir, ".json")
            count = sum(1 for _, h in file_hashes if h in cached_keys)
            print(f"  [{i}] {ldir.name} ({count}/{len(file_hashes)} cached)", file=sys.stderr)
        try:
            idx = int(input("Pick one [0]: ").strip() or "0")
            chosen = sorted(label_dirs)[idx]
        except (ValueError, IndexError):
            print("Invalid selection.", file=sys.stderr)
            return

    # Collect labels
    cached_keys = list_cached_keys(chosen, ".json")
    exported = 0
    skipped = 0
    lines = ["# Generated by semantic-navigator", f"# Model identity: {chosen.name}", ""]

    for path, h in file_hashes:
        if h not in cached_keys:
            skipped += 1
            continue
        label = load_cached_label(chosen, h)
        if label is None:
            skipped += 1
            continue
        lines.append(f'["{path}"]')
        lines.append(f'overarchingTheme = "{label.overarchingTheme}"')
        lines.append(f'distinguishingFeature = "{label.distinguishingFeature}"')
        lines.append(f'label = "{label.label}"')
        lines.append("")
        exported += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Exported {exported} labels to {output_path}.")
    if skipped:
        print(f"Skipped {skipped} files without cached labels.")


def _flush_labels(repository: str):
    """Delete cached labels only (keeps embeddings)."""
    repo_dir = repo_cache_dir(repository)
    labels_dir = repo_dir / "labels"
    if labels_dir.exists():
        size = sum(f.stat().st_size for f in labels_dir.rglob("*") if f.is_file())
        shutil.rmtree(labels_dir)
        print(f"Flushed label cache for {repository} ({size / 1e6:.1f} MB)")
    else:
        print(f"No label cache found for {repository}")


def _export_clusters(repository: str, output_path: str, embedding_model: str, gpu: bool, batch_size: int | None, openai_embedding_model: str | None):
    """Run embed + cluster, export to JSON for external labeling."""
    import json
    import time
    from semantic_navigator.models import AspectPool, Facets
    from fastembed import TextEmbedding

    embedding_model_name = openai_embedding_model or embedding_model

    # Minimal Facets — no LLM backend needed
    if openai_embedding_model:
        import openai
        openai_client = openai.AsyncOpenAI()
        emb_model = None
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        emb_model = TextEmbedding(model_name=embedding_model, providers=providers)
        openai_client = None

    facets = Facets(
        repository=repository,
        model_identity="claude-code-skill",
        pool=AspectPool([]),
        embedding_model=emb_model,
        embedding_model_name=embedding_model_name,
        gpu=gpu,
        batch_size=batch_size or 256,
        timeout=60,
        debug=False,
        openai_client=openai_client,
        openai_embedding_model=openai_embedding_model,
    )

    async def run():
        t0 = time.monotonic()
        initial_cluster = await embed(facets, repository)
        print(f"Embedded {len(initial_cluster.embeds)} files in {time.monotonic() - t0:.1f}s")

        t0 = time.monotonic()
        ct = build_cluster_tree(initial_cluster)
        print(f"Clustered in {time.monotonic() - t0:.1f}s")

        data = export_cluster_tree(facets, ct)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Exported to {output_path}")

    asyncio.run(run())


def _import_labels(arguments: argparse.Namespace, remaining: list[str]):
    """Import labels from JSON, populate cache, then launch TUI normally."""
    import json
    from fastembed import TextEmbedding
    from semantic_navigator.models import AspectPool, Facets

    with open(arguments.import_labels, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_identity = data.get("model_identity") or "claude-code-skill"

    embedding_model_name = arguments.embedding_model
    openai_emb_model = None
    openai_client = None
    if arguments.openai:
        openai_emb_model = arguments.openai_embedding_model or "text-embedding-3-small"
        if openai_emb_model == "local":
            openai_emb_model = None
        if openai_emb_model:
            embedding_model_name = openai_emb_model
            import openai
            openai_client = openai.AsyncOpenAI()

    # Set up embedding model for the re-embed step
    if openai_client is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if arguments.gpu else ["CPUExecutionProvider"]
        emb_model = TextEmbedding(model_name=arguments.embedding_model, providers=providers)
    else:
        emb_model = None

    facets = Facets(
        repository=arguments.repository,
        model_identity=model_identity,
        pool=AspectPool([]),
        embedding_model=emb_model,
        embedding_model_name=embedding_model_name,
        gpu=arguments.gpu,
        batch_size=arguments.batch_size or 256,
        timeout=arguments.timeout,
        debug=arguments.debug,
        openai_client=openai_client,
        openai_embedding_model=openai_emb_model,
    )

    import_labels(facets, data)

    # Run the normal pipeline — labels are cached, labeling will be skipped
    # But embed+cluster still needed to build the tree structure
    async def run():
        timings: dict[str, float] = {}
        initial_cluster = await embed(facets, arguments.repository)
        print(f"Processing {len(initial_cluster.embeds)} files...")
        return await tree(facets, arguments.repository, initial_cluster, timings)

    try:
        tree_ = asyncio.run(run())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return

    UI(tree_).run()


def _parse_cli_tool(remaining: list[str], parser: argparse.ArgumentParser) -> list[str] | None:
    """Parse CLI tool from remaining args. Returns command list or None."""
    if not remaining or not remaining[0].startswith("--"):
        return None
    tool_name = remaining[0][2:]
    if shutil.which(tool_name) is None:
        parser.error(f"CLI tool '{tool_name}' not found on PATH")
    return [tool_name] + remaining[1:]


def _build_model_identity(arguments: argparse.Namespace, cli_command: list[str] | None) -> str:
    """Build model identity string from arguments."""
    identity_parts = []
    if arguments.local:
        identity_parts.append(f"local:{arguments.local}")
        if arguments.local_file:
            identity_parts.append(f"file:{arguments.local_file}")
    if cli_command:
        identity_parts.append(f"cli:{shlex.join(cli_command)}")
    if arguments.openai:
        identity_parts.append(f"openai:{arguments.completion_model}")
    return "+".join(identity_parts)


def main():
    parser = argparse.ArgumentParser(
        prog = "semantic-navigator",
        description = "Cluster documents by semantic facets",
    )

    parser.add_argument("repository", nargs = "?")
    parser.add_argument("--embedding-model", default = "BAAI/bge-large-en-v1.5")
    parser.add_argument("--gpu", action = "store_true")
    parser.add_argument("--cpu", action = "store_true", help = "Add a CPU local model worker alongside GPU workers")
    parser.add_argument("--cpu-offload", action = "store_true")
    parser.add_argument("--device", type = str, default = "0", help = "GPU devices as id:layers pairs, comma-separated (e.g. 0:-1,1:30). Layers optional — omit for auto.")
    parser.add_argument("--gpu-layers", type = int, default = None)
    parser.add_argument("--batch-size", type = int, default = None)
    parser.add_argument("--concurrency", type = int, default = None)
    parser.add_argument("--timeout", type = int, default = 60)
    parser.add_argument("--list-devices", action = "store_true")
    parser.add_argument("--flush-cache", action = "store_true", help = "Delete cached labels for the given repository (preserves embeddings)")
    parser.add_argument("--flush-labels", action = "store_true", help = "Delete cached labels only (keeps embeddings)")
    parser.add_argument("--status", action = "store_true", help = "Show cache status for the repository")
    parser.add_argument("--export-toml", type = str, default = None, metavar = "FILE", help = "Export cached labels as TOML to the given file")
    parser.add_argument("--export-clusters", type = str, default = None, metavar = "FILE", help = "Export clusters as JSON for external labeling (no LLM needed)")
    parser.add_argument("--import-labels", type = str, default = None, metavar = "FILE", help = "Import labels from JSON, populate cache, and launch TUI")
    parser.add_argument("--erase-models", action = "store_true", help = "Delete downloaded HuggingFace models")
    parser.add_argument("--openai", action = "store_true", help = "Use OpenAI API for labeling (requires OPENAI_API_KEY)")
    parser.add_argument("--completion-model", default = "gpt-4o-mini", help = "OpenAI model for labeling (default: gpt-4o-mini)")
    parser.add_argument("--openai-embedding-model", default = None, help = "OpenAI embedding model (default: text-embedding-3-small with --openai; 'local' to force fastembed)")
    parser.add_argument("--local", default = None)
    parser.add_argument("--local-file", default = None)
    parser.add_argument("--n-ctx", type = int, default = None)
    parser.add_argument("--debug", action = "store_true")
    arguments, remaining = parser.parse_known_args()

    if arguments.list_devices:
        list_devices()
        return

    if arguments.erase_models:
        _handle_erase_models()
        return

    if arguments.repository is None:
        parser.error("the following arguments are required: repository")

    if arguments.flush_cache:
        _flush_cache(arguments.repository)
        return

    if arguments.flush_labels:
        _flush_labels(arguments.repository)
        return

    if arguments.status:
        _show_status(arguments.repository)
        return

    if arguments.export_toml is not None:
        _export_toml(arguments.repository, arguments.export_toml)
        return

    if arguments.export_clusters is not None:
        _export_clusters(arguments.repository, arguments.export_clusters, arguments.embedding_model, arguments.gpu, arguments.batch_size, arguments.openai_embedding_model if arguments.openai else None)
        return

    if arguments.import_labels is not None:
        _import_labels(arguments, remaining)
        return

    cli_command = _parse_cli_tool(remaining, parser)
    has_cli_tool = cli_command is not None

    if arguments.local is None and not has_cli_tool and not arguments.openai:
        parser.error("no backend specified (e.g. --openai, --gemini, --llm, or --local)")

    if arguments.cpu_offload and not arguments.gpu:
        parser.error("--cpu-offload requires --gpu")

    if arguments.cpu and not arguments.gpu:
        parser.error("--cpu only makes sense with --gpu (without --gpu, CPU is already the default)")

    if arguments.gpu_layers is not None and arguments.local is None:
        parser.error("--gpu-layers requires --local")

    if arguments.local_file is not None and arguments.local is None:
        parser.error("--local-file requires --local")

    if arguments.n_ctx is not None and arguments.local is None:
        parser.error("--n-ctx requires --local")

    if arguments.concurrency is not None and arguments.local is not None and not has_cli_tool and not arguments.openai:
        parser.error("--concurrency has no effect with --local only (local model concurrency is 1 per device)")

    try:
        devices: list[tuple[int, int | None]] = []
        for part in arguments.device.split(","):
            part = part.strip()
            if ":" in part:
                dev_str, layers_str = part.split(":", 1)
                devices.append((int(dev_str), int(layers_str)))
            else:
                devices.append((int(part), None))
    except ValueError:
        parser.error(f"--device must be comma-separated id[:layers] pairs, got: {arguments.device}")

    if arguments.n_ctx is None:
        arguments.n_ctx = 8192
    if arguments.concurrency is None:
        arguments.concurrency = 4

    model_identity = _build_model_identity(arguments, cli_command)
    openai_model = arguments.completion_model if arguments.openai else None

    # When using --openai, default to OpenAI embeddings unless overridden with 'local'
    if arguments.openai and arguments.openai_embedding_model is None:
        arguments.openai_embedding_model = "text-embedding-3-small"
    if arguments.openai_embedding_model == "local":
        arguments.openai_embedding_model = None
    embedding_model_name = arguments.embedding_model
    if arguments.openai_embedding_model:
        embedding_model_name = arguments.openai_embedding_model

    facets = initialize(arguments.repository, model_identity, cli_command, arguments.local, arguments.local_file, embedding_model_name, arguments.gpu, arguments.cpu, arguments.cpu_offload, devices, arguments.gpu_layers, arguments.batch_size, arguments.concurrency, arguments.n_ctx, arguments.timeout, arguments.debug, openai_model=openai_model, openai_embedding_model=arguments.openai_embedding_model)

    async def async_tasks():
        timings: dict[str, float] = {}
        total_start = time.monotonic()

        with timed("Reading & embedding", timings):
            initial_cluster = await embed(facets, arguments.repository)

        print(f"Processing {len(initial_cluster.embeds)} files...")
        tree_ = await tree(facets, arguments.repository, initial_cluster, timings)

        total = time.monotonic() - total_start
        parts = " | ".join(f"{k}: {_fmt_time(v)}" for k, v in timings.items())
        print(f"Done! Total: {_fmt_time(total)} ({parts})")
        return tree_

    try:
        tree_ = asyncio.run(async_tasks())
    except KeyboardInterrupt:
        print("\nInterrupted. Progress has been cached and will resume on next run.")
        return

    UI(tree_).run()

if __name__ == "__main__":
    main()
