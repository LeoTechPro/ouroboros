#!/usr/bin/env python3
"""The real managed Node/npm toolchain behind the Claudexor CLI, on one platform.

PROVES — with disposable HOME/app/data/config roots, no ``node``/``npm``/``npx``/
``corepack`` reachable on PATH, no ambient npm configuration and no Claudexor
harness-binary override (``CLAUDEXOR_CODEX_BIN`` and friends), that the production
``ClaudexorRuntimeManager.ensure_cli_command()`` downloads the exact pinned
official Node archive (``.zip`` on Windows, ``.tar.gz`` elsewhere) and Claudexor
closure, extracts Node plus its regular-file npm tree, verifies and promotes both,
and returns ``[managed node, CLI]``. The managed Node then runs the returned CLI,
and runs the extracted ``npm-cli.js`` directly — never an npm ``.cmd``/``.ps1``
shim — to pack and globally install a local package offline into a disposable
prefix, whose entry point it runs. A fresh manager selects the same command
without re-promoting, and a deleted npm entrypoint is repaired from the verified
archive cache.

DOES NOT PROVE — that any vendor harness installs, logs in or runs.
``harness install <harness> --target local`` is never invoked: the pinned engine
refuses local installs on Windows, a vendor package download is not a toolchain
fact, and no account exists in CI. This is the Ouroboros half of that path — the
managed toolchain ``claudexor_daemon.install_missing_harness_cli`` hands to the
engine — and nothing more.

Usage:
    python -I scripts/claudexor_toolchain_smoke.py [--root EMPTY_DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AMBIENT_TOOLS = ("node", "npm", "npx", "corepack")
# Beyond scrub_environment's owner/credential/process-override set: engine homes and
# harness-binary overrides, and ambient npm/Node configuration the runner may carry.
DROPPED_PREFIXES = ("CLAUDEXOR_", "NPM_CONFIG_", "NODE_", "COREPACK_")
STEP_TIMEOUT_SEC = 180
WITNESS_PACKAGE = "ouroboros-toolchain-witness"
WITNESS_TOKEN = "managed-npm-install-ok"
NPM_QUIET = ("--offline", "--ignore-scripts", "--no-audit", "--no-fund",
             "--no-update-notifier", "--loglevel=error")


class WitnessFailure(RuntimeError):
    """A named refusal; every exit path that is not success carries one."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)


def scrubbed_path(value: str) -> Tuple[str, List[str]]:
    """Drop every PATH entry that provides Node or an npm-family launcher."""
    kept: List[str] = []
    dropped: List[str] = []
    for entry in value.split(os.pathsep):
        if entry and any(shutil.which(tool, path=entry) for tool in AMBIENT_TOOLS):
            dropped.append(entry)
        else:
            kept.append(entry)
    return os.pathsep.join(kept), dropped


def isolate_environment(root: pathlib.Path) -> List[str]:
    """Rebind this process before any Ouroboros config import.

    Only HOME-shaped roots are set: the app and data roots then derive from the
    disposable home exactly as production derives them from a real one. Returns
    the PATH entries that were dropped.
    """
    from ouroboros.test_environment import scrub_environment

    home = root / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = {
        key: value for key, value in scrub_environment(os.environ).items()
        if not key.upper().startswith(DROPPED_PREFIXES)
    }
    env["PATH"], dropped = scrubbed_path(env.get("PATH", ""))
    env.update(
        HOME=str(home), USERPROFILE=str(home),
        APPDATA=str(home / "AppData" / "Roaming"), LOCALAPPDATA=str(home / "AppData" / "Local"),
        XDG_CONFIG_HOME=str(home / ".config"), XDG_CACHE_HOME=str(home / ".cache"),
    )
    os.environ.clear()
    os.environ.update(env)
    return dropped


def _run(argv: Sequence[Any], code: str, *, cwd: Optional[pathlib.Path] = None,
         env: Optional[Dict[str, str]] = None) -> str:
    command = [str(part) for part in argv]
    try:
        completed = subprocess.run(
            command, cwd=str(cwd) if cwd else None, env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=STEP_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WitnessFailure(code, f"{command[1:3]} did not complete: {type(exc).__name__}: {exc}") from exc
    if completed.returncode != 0:
        raise WitnessFailure(
            code, f"{command[1:3]} exited {completed.returncode}: "
            f"{(completed.stderr or completed.stdout)[-2000:].strip()}",
        )
    return completed.stdout.strip()


def official_npm_cli(node: pathlib.Path) -> pathlib.Path:
    """Where the official Node distribution keeps npm relative to its executable."""
    if node.name.lower() == "node.exe":
        return node.parent / "node_modules" / "npm" / "bin" / "npm-cli.js"
    return node.parent.parent / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js"


def npm_install_witness(node: pathlib.Path, npm_cli: pathlib.Path, work: pathlib.Path) -> Dict[str, Any]:
    """Pack and globally install a local package with the managed npm, offline."""
    package = work / "package"
    package.mkdir(parents=True)
    (package / "package.json").write_text(json.dumps({
        "name": WITNESS_PACKAGE, "version": "1.0.0", "bin": {WITNESS_PACKAGE: "cli.js"},
    }), encoding="utf-8")
    (package / "cli.js").write_text(
        f"#!/usr/bin/env node\nconsole.log({json.dumps(WITNESS_TOKEN)});\n", encoding="utf-8")
    npm = [node, npm_cli]
    try:
        packed = json.loads(_run([*npm, "pack", "--json", "--pack-destination", work, *NPM_QUIET],
                                 "npm_pack_failed", cwd=package))
        tarball = work / str(packed[0]["filename"])
    except (ValueError, LookupError, TypeError) as exc:
        raise WitnessFailure("npm_pack_failed", f"npm pack returned no tarball: {exc}") from exc
    prefix = work / "prefix"
    _run([*npm, "install", "--global", "--prefix", prefix, *NPM_QUIET, tarball],
         "npm_install_failed", cwd=work)
    global_root = pathlib.Path(_run([*npm, "root", "--global", "--prefix", prefix],
                                    "npm_root_failed", cwd=work))
    output = _run([node, global_root / WITNESS_PACKAGE / "cli.js"], "npm_installed_entry_failed", cwd=work)
    if output != WITNESS_TOKEN:
        raise WitnessFailure("npm_installed_entry_failed", f"installed entry printed {output!r}")
    return {"tarball": tarball.name, "global_root": str(global_root)}


def run_witness(root: pathlib.Path, dropped_path: List[str]) -> Dict[str, Any]:
    facts: Dict[str, Any] = {"isolated_root": str(root), "path_entries_dropped": dropped_path}
    ambient = {tool: found for tool in AMBIENT_TOOLS if (found := shutil.which(tool))}
    if ambient:
        raise WitnessFailure("ambient_node_present", f"PATH still provides {ambient}")
    overrides = sorted(key for key in os.environ if key.upper().startswith(DROPPED_PREFIXES)
                       or key.upper() == "OUROBOROS_CLAUDEXOR_BIN")
    if overrides:
        raise WitnessFailure("ambient_override_present", f"environment still carries {overrides}")

    from ouroboros.claudexor_daemon import owned_config_dir
    from ouroboros.claudexor_runtime import (
        ClaudexorRuntimeError,
        ClaudexorRuntimeManager,
        managed_node_dir,
        managed_runtime_dir,
    )
    from ouroboros.config import DATA_DIR
    from ouroboros.platform_layer import embedded_node_candidates, node_distribution_platform

    data_dir = pathlib.Path(DATA_DIR).resolve()
    if root not in data_dir.parents:
        raise WitnessFailure("data_root_not_isolated", f"DATA_DIR {data_dir} is outside {root}")
    manager = ClaudexorRuntimeManager()
    pin, platform_key = manager.pin, node_distribution_platform()
    if pin is None or pin.cli_entrypoint is None:
        raise WitnessFailure("runtime_cli_unpinned", "this checkout pins no managed Claudexor CLI")
    facts.update(engine_pin=pin.version, node_pin=pin.node_version, platform=platform_key,
                 data_dir=str(data_dir))
    try:
        command = manager.ensure_cli_command()
    except ClaudexorRuntimeError as exc:
        raise WitnessFailure(exc.code, str(exc)) from exc
    facts["node_archive"] = pin.node_artifacts[platform_key].archive_name
    node_root = managed_node_dir(pin, platform_key)
    node = embedded_node_candidates(node_root)[0]
    if len(command) != 2 or pathlib.Path(command[0]) != node:
        raise WitnessFailure("cli_command_unexpected", f"{command} does not run the managed Node {node}")
    cli, npm_cli = pathlib.Path(command[1]), official_npm_cli(node)
    metadata_path = node_root / "managed-node.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2 or not str(metadata.get("archive_npm_cli") or "").endswith(
            "node_modules/npm/bin/npm-cli.js") or not npm_cli.is_file():
        raise WitnessFailure("npm_tree_missing", f"managed Node metadata {metadata} or {npm_cli} is absent")
    runtime_meta = json.loads((managed_runtime_dir(pin) / "managed-runtime.json").read_text(encoding="utf-8"))
    facts.update(managed_node=str(node), managed_cli=str(cli), npm_cli=str(npm_cli),
                 node_metadata_schema=metadata["schema_version"],
                 runtime_archive_source=runtime_meta.get("archive_source"))

    node_version = _run([node, "--version"], "managed_node_failed")
    if node_version != f"v{pin.node_version}":
        raise WitnessFailure("managed_node_failed", f"managed Node reports {node_version!r}")
    # The same data-plane binding install_missing_harness_cli gives the CLI.
    cli_env = dict(os.environ, CLAUDEXOR_CONFIG_DIR=str(owned_config_dir()))
    cli_version = _run([node, cli, "--version"], "managed_cli_failed", env=cli_env)
    if pin.version not in cli_version:
        raise WitnessFailure("managed_cli_failed", f"managed CLI reports {cli_version!r}")
    npm_version = _run([node, npm_cli, "--version"], "managed_npm_failed")
    facts.update(node_version=node_version, cli_version=cli_version, npm_version=npm_version)
    facts["npm_install"] = npm_install_witness(node, npm_cli, root / "npm-witness")

    before = metadata_path.stat()
    if (ClaudexorRuntimeManager().ensure_cli_command() != command
            or ClaudexorRuntimeManager().resolve_cli_command() != command
            or metadata_path.stat().st_mtime_ns != before.st_mtime_ns):
        raise WitnessFailure("reselect_unstable", "a fresh manager did not reselect the installed toolchain")
    npm_cli.unlink()
    try:
        repaired = ClaudexorRuntimeManager().ensure_cli_command()
    except ClaudexorRuntimeError as exc:
        raise WitnessFailure(exc.code, f"repair failed: {exc}") from exc
    if repaired != command or _run([node, npm_cli, "--version"], "repair_failed") != npm_version:
        raise WitnessFailure("repair_failed", "a deleted npm entrypoint was not restored")
    facts["repair"] = "deleted npm-cli.js restored from the verified archive cache"
    return facts


LIMITS = (
    "### What this check does NOT cover",
    "",
    "- **No vendor harness was installed, logged in or run.** `harness install --target "
    "local` is not invoked; the pinned engine refuses local installs on Windows. A green "
    "row says the managed Node/npm toolchain works here, not that Codex or Claude does.",
    "- No daemon, model, account or credential was involved.",
    "- npm was proven on an offline local-tarball install; registry access is not tested.",
)


def emit_summary(facts: Dict[str, Any], verdict: str, detail: str) -> None:
    lines = [
        f"## Claudexor CLI toolchain — `{sys.platform}`",
        "",
        f"**{verdict}** — {detail}",
        "",
        "| fact | value |",
        "| --- | --- |",
        *(f"| `{key}` | {json.dumps(value) if isinstance(value, (dict, list)) else value} |"
          for key, value in sorted(facts.items())),
        "",
        *LIMITS,
        "",
    ]
    text = "\n".join(lines)
    print(text, flush=True)
    target = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if target:
        try:
            with open(target, "a", encoding="utf-8") as stream:
                stream.write(text + "\n")
        except OSError as exc:  # reporting never masks the verdict
            print(f"[toolchain] could not write GITHUB_STEP_SUMMARY: {exc}", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=pathlib.Path, default=None,
                        help="empty or absent disposable directory (default: a new temp dir)")
    args = parser.parse_args(argv)
    root = args.root or pathlib.Path(tempfile.mkdtemp(prefix="cx-toolchain-"))
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    if any(root.iterdir()):
        # A pre-seeded cache or install would turn "downloaded and promoted" into a claim.
        parser.error(f"--root must be empty: {root}")
    dropped = isolate_environment(root)
    try:
        facts = run_witness(root, dropped)
    except WitnessFailure as exc:
        emit_summary({"refusal_code": exc.code}, "FAILED", f"`{exc.code}` — {exc}")
        return 1
    except Exception as exc:  # unexpected: still loud, still named (a runtime error keeps its code)
        code = str(getattr(exc, "code", "") or "unexpected_error")
        emit_summary({"refusal_code": code}, "FAILED", f"`{code}` — {type(exc).__name__}: {exc}")
        raise
    emit_summary(facts, "PASSED", "the pinned managed Node/npm toolchain was installed from "
                 "official archives and exercised with no ambient Node or npm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
