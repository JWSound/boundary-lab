"""Exercise the actual batch updater against disposable Git repositories."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows batch updater")
ROOT = Path(__file__).resolve().parents[1]


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


@pytest.fixture
def checkout(tmp_path):
    remote = tmp_path / "release origin"
    remote.mkdir()
    git(remote, "init", "-b", "main")
    git(remote, "config", "user.name", "Updater test")
    git(remote, "config", "user.email", "updater@example.invalid")
    (remote / "tracked.txt").write_text("stable")
    git(remote, "add", ".")
    git(remote, "commit", "-m", "stable")
    git(remote, "tag", "v1.0.0")
    stable = git(remote, "rev-parse", "HEAD")
    (remote / "tracked.txt").write_text("development")
    git(remote, "commit", "-am", "development")
    local = tmp_path / "user checkout"
    subprocess.run(["git", "clone", str(remote), str(local)], check=True, capture_output=True)
    git(local, "config", "user.name", "Updater test")
    git(local, "config", "user.email", "updater@example.invalid")
    source = (ROOT / "01_install_update_boundary-lab.bat").read_text()
    body = source[source.index("\n:UPDATE_STABLE\n") + 1 :]
    harness = tmp_path / "update-test.bat"
    harness.write_text(
        "@echo off\nsetlocal EnableExtensions DisableDelayedExpansion\n"
        "call :UPDATE_STABLE\nexit /b %errorlevel%\n"
        ":ENSURE_GIT\nexit /b 0\n"
        ':RESOLVE_STABLE_TAG\nset "LATEST_RELEASE=v1.0.0"\nexit /b 0\n' + body
    )

    def update():
        env = {**os.environ, "PROJECT_DIR": str(local), "REPO_URL": str(remote)}
        return subprocess.run(["cmd.exe", "/d", "/c", str(harness)], env=env, capture_output=True, text=True)

    return local, stable, update


def test_selects_published_tag_instead_of_development_tip(checkout):
    local, stable, update = checkout
    (local / "user-project.blab.json").write_text("user data")
    result = update()
    assert result.returncode == 0, result.stdout + result.stderr
    assert git(local, "rev-parse", "HEAD") == stable
    assert git(local, "branch", "--show-current") == ""
    assert (local / "user-project.blab.json").read_text() == "user data"
    assert update().returncode == 0  # Detached installations remain updateable.


@pytest.mark.parametrize("state", ["dirty", "staged", "feature", "unpublished", "detached-unpublished"])
def test_preserves_developer_work(checkout, state):
    local, stable, update = checkout
    if state == "feature":
        git(local, "switch", "-c", "feature/work")
    else:
        (local / "tracked.txt").write_text("local work")
        if state != "dirty":
            git(local, "add", ".")
        if "unpublished" in state:
            git(local, "commit", "-m", "local work")
            if state == "detached-unpublished":
                git(local, "checkout", "--detach")
    before = git(local, "rev-parse", "HEAD")
    result = update()
    assert result.returncode != 0
    assert git(local, "rev-parse", "HEAD") == before
    if state != "feature":
        assert (local / "tracked.txt").read_text() == "local work"
