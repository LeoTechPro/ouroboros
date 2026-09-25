"""The managed CLI toolchain witness's own logic, under the ordinary suite.

Only what needs no network: the PATH scrub, the process rebinding done before the
runtime is imported, the refusals that precede any install, and the CI wiring. The
download, extraction and npm runs are what the `toolchain` CI job exercises for real;
repeating them here would mean mocking the thing under test.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest
import yaml

from ouroboros.claudexor_runtime import ClaudexorRuntimeManager

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_witness():
    script = REPO_ROOT / "scripts" / "claudexor_toolchain_smoke.py"
    spec = importlib.util.spec_from_file_location("claudexor_toolchain_smoke", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


witness = _load_witness()


def _tool(directory: pathlib.Path, name: str) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (f"{name}.cmd" if os.name == "nt" else name)
    path.write_text("", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_path_scrub_drops_every_node_and_npm_provider(tmp_path):
    node_dir, npm_dir, clean = tmp_path / "node", tmp_path / "npm", tmp_path / "clean"
    _tool(node_dir, "node")
    _tool(npm_dir, "npx")
    _tool(clean, "git")
    value = os.pathsep.join([str(node_dir), str(clean), str(npm_dir)])

    kept, dropped = witness.scrubbed_path(value)

    assert kept == str(clean)
    assert dropped == [str(node_dir), str(npm_dir)]


def test_isolation_rebinds_home_and_drops_overrides_before_runtime_imports(tmp_path, monkeypatch):
    node_dir, clean = tmp_path / "ambient-node", tmp_path / "clean"
    _tool(node_dir, "npm")
    clean.mkdir()
    monkeypatch.setattr(witness.os, "environ", {
        "PATH": os.pathsep.join([str(node_dir), str(clean)]),
        "HOME": str(tmp_path / "operator"),
        "OUROBOROS_DATA_DIR": str(tmp_path / "live-data"),
        "OUROBOROS_BUNDLE_DIR": str(tmp_path / "bundle"),
        "CLAUDEXOR_CONFIG_DIR": str(tmp_path / "live-engine"),
        "CLAUDEXOR_CODEX_BIN": str(tmp_path / "codex.exe"),
        "npm_config_prefix": str(tmp_path / "npm-prefix"),
        "NODE_OPTIONS": "--require=hook.js",
        "OPENAI_API_KEY": "fixture-secret",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
    })
    root = tmp_path / "isolated"

    dropped = witness.isolate_environment(root)

    env = witness.os.environ
    assert dropped == [str(node_dir)] and env["PATH"] == str(clean)
    assert env["HOME"] == env["USERPROFILE"] == str(root / "home")
    assert env["LOCALAPPDATA"].startswith(str(root / "home"))
    # No Ouroboros root is set: production derives app/data from the disposable home.
    assert not [key for key in env if key.startswith(("OUROBOROS_", "CLAUDEXOR_"))]
    assert not {"npm_config_prefix", "NODE_OPTIONS", "OPENAI_API_KEY"} & set(env)
    assert env["GITHUB_STEP_SUMMARY"] == str(tmp_path / "summary.md")


@pytest.mark.parametrize("ambient", ("path", "override"))
def test_witness_refuses_ambient_node_or_override_before_the_runtime(tmp_path, monkeypatch, ambient):
    node_dir = tmp_path / "ambient"
    _tool(node_dir, "node")
    env = {"PATH": str(node_dir) if ambient == "path" else str(tmp_path)}
    if ambient == "override":
        env["CLAUDEXOR_CODEX_BIN"] = str(tmp_path / "codex.exe")
    monkeypatch.setattr(witness.os, "environ", env)

    with pytest.raises(witness.WitnessFailure) as excinfo:
        witness.run_witness(tmp_path, [])

    assert excinfo.value.code == (
        "ambient_node_present" if ambient == "path" else "ambient_override_present")


def test_witness_expects_npm_where_the_official_distribution_and_manager_keep_it():
    for node in (pathlib.Path("cx/node-standalone/node.exe"), pathlib.Path("cx/node-standalone/bin/node")):
        assert witness.official_npm_cli(node) == ClaudexorRuntimeManager._managed_npm_cli(node)
    assert witness.official_npm_cli(pathlib.Path("n/node.exe")).parent.parent.parent == pathlib.Path(
        "n/node_modules")


def test_main_requires_an_empty_root(tmp_path):
    (tmp_path / "stale-cache").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        witness.main(["--root", str(tmp_path)])


def test_isolated_start_ignores_ambient_pythonpath_before_script_scrub(tmp_path):
    injected = tmp_path / "injected"
    injected.mkdir()
    marker = tmp_path / "sitecustomize-ran"
    (injected / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n", encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, "-I", str(REPO_ROOT / "scripts" / "claudexor_toolchain_smoke.py"), "--help"],
        env={**os.environ, "PYTHONPATH": str(injected)},
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--root" in result.stdout
    assert not marker.exists()


def test_a_refusal_is_named_and_the_limits_always_reach_the_summary(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(witness.os, "environ", {"GITHUB_STEP_SUMMARY": str(summary)})
    monkeypatch.setattr(witness, "isolate_environment", lambda _root: [])

    def refuse(_root, _dropped):
        raise witness.WitnessFailure("runtime_node_archive_invalid", "fixture refusal")

    monkeypatch.setattr(witness, "run_witness", refuse)

    assert witness.main(["--root", str(tmp_path / "root")]) == 1
    written = summary.read_text(encoding="utf-8")
    assert "FAILED" in written and "`runtime_node_archive_invalid`" in written
    assert "No vendor harness was installed" in written


def test_platform_gate_runs_the_toolchain_witness_on_windows_pull_requests():
    path = REPO_ROOT / ".github" / "workflows" / "claudexor-platform-gate.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    for event in ("push", "pull_request"):
        assert "scripts/claudexor_toolchain_smoke.py" in triggers[event]["paths"]
    assert triggers["pull_request"]["branches"] == ["ouroboros"]
    job = workflow["jobs"]["toolchain"]
    assert "if" not in job
    assert "windows-latest" in job["strategy"]["matrix"]["os"]
    steps = job["steps"]
    assert not [step for step in steps if "setup-node" in str(step.get("uses", ""))]
    commands = "\n".join(str(step.get("run", "")) for step in steps)
    assert "python -I scripts/claudexor_toolchain_smoke.py --root" in commands
    # The witness proves the toolchain only; it neither pre-seeds a harness nor claims one.
    assert "CLAUDEXOR_CODEX_BIN" not in commands and "harness install" not in commands
    assert "npm " not in commands
