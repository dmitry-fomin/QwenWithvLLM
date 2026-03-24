"""Тесты для shell-скриптов (проверка синтаксиса и содержимого)."""

import os
import subprocess
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def get_script_files():
    return list(SCRIPTS_DIR.glob("*.sh"))


class TestScriptsSyntax:
    @pytest.mark.parametrize("script", get_script_files(), ids=lambda p: p.name)
    def test_script_exists_and_not_empty(self, script):
        assert script.exists()
        assert script.stat().st_size > 0

    @pytest.mark.parametrize("script", get_script_files(), ids=lambda p: p.name)
    def test_script_is_executable(self, script):
        assert os.access(script, os.X_OK), f"{script.name} is not executable"

    @pytest.mark.parametrize("script", get_script_files(), ids=lambda p: p.name)
    def test_script_has_shebang(self, script):
        first_line = script.read_text().splitlines()[0]
        assert first_line.startswith("#!/bin/bash"), (
            f"{script.name}: missing #!/bin/bash shebang"
        )

    @pytest.mark.parametrize("script", get_script_files(), ids=lambda p: p.name)
    def test_script_bash_syntax(self, script):
        """Проверяет синтаксис bash через bash -n."""
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"{script.name} syntax error:\n{result.stderr}"
        )


class TestStartVllmScript:
    def test_contains_vllm_entrypoint(self):
        content = (SCRIPTS_DIR / "start_vllm.sh").read_text()
        assert "vllm.entrypoints.openai.api_server" in content

    def test_reads_config_file(self):
        content = (SCRIPTS_DIR / "start_vllm.sh").read_text()
        assert "source" in content

    def test_exposes_port_8000(self):
        content = (SCRIPTS_DIR / "start_vllm.sh").read_text()
        assert "8000" in content


class TestSwitchModelScript:
    def test_kills_existing_process(self):
        content = (SCRIPTS_DIR / "switch_model.sh").read_text()
        assert "pkill" in content

    def test_starts_new_model(self):
        content = (SCRIPTS_DIR / "switch_model.sh").read_text()
        assert "start_vllm.sh" in content


class TestDownloadModelScript:
    def test_uses_snapshot_download(self):
        content = (SCRIPTS_DIR / "download_model.sh").read_text()
        assert "snapshot_download" in content

    def test_supports_resume(self):
        content = (SCRIPTS_DIR / "download_model.sh").read_text()
        assert "resume_download" in content
