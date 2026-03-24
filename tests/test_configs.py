"""Тесты для конфигурационных файлов моделей."""

import os
from pathlib import Path

import pytest


CONFIGS_DIR = Path(__file__).parent.parent / "configs"
REQUIRED_KEYS = [
    "MODEL_PATH",
    "MODEL_NAME",
    "TENSOR_PARALLEL_SIZE",
    "MAX_MODEL_LEN",
    "GPU_MEMORY_UTILIZATION",
    "DTYPE",
]


def parse_env_file(path: Path) -> dict:
    """Парсит .env файл в словарь, пропуская комментарии и пустые строки."""
    result = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip().strip('"')
    return result


def get_config_files():
    """Возвращает все .env файлы из configs/."""
    return list(CONFIGS_DIR.glob("*.env"))


class TestConfigFiles:
    def test_configs_directory_exists(self):
        assert CONFIGS_DIR.exists(), f"Directory {CONFIGS_DIR} not found"

    def test_at_least_one_config_exists(self):
        configs = get_config_files()
        assert len(configs) > 0, "No .env config files found"

    @pytest.mark.parametrize("config_path", get_config_files(), ids=lambda p: p.name)
    def test_config_has_required_keys(self, config_path):
        config = parse_env_file(config_path)
        for key in REQUIRED_KEYS:
            assert key in config, f"{config_path.name}: missing required key '{key}'"

    @pytest.mark.parametrize("config_path", get_config_files(), ids=lambda p: p.name)
    def test_tensor_parallel_is_positive_int(self, config_path):
        config = parse_env_file(config_path)
        tp = int(config["TENSOR_PARALLEL_SIZE"])
        assert tp > 0, f"{config_path.name}: TENSOR_PARALLEL_SIZE must be > 0"

    @pytest.mark.parametrize("config_path", get_config_files(), ids=lambda p: p.name)
    def test_max_model_len_is_positive_int(self, config_path):
        config = parse_env_file(config_path)
        mml = int(config["MAX_MODEL_LEN"])
        assert mml > 0, f"{config_path.name}: MAX_MODEL_LEN must be > 0"

    @pytest.mark.parametrize("config_path", get_config_files(), ids=lambda p: p.name)
    def test_gpu_memory_utilization_in_range(self, config_path):
        config = parse_env_file(config_path)
        gmu = float(config["GPU_MEMORY_UTILIZATION"])
        assert 0.0 < gmu <= 1.0, f"{config_path.name}: GPU_MEMORY_UTILIZATION must be in (0, 1]"

    @pytest.mark.parametrize("config_path", get_config_files(), ids=lambda p: p.name)
    def test_dtype_is_valid(self, config_path):
        config = parse_env_file(config_path)
        valid_dtypes = {"bfloat16", "float16", "float32", "auto"}
        assert config["DTYPE"] in valid_dtypes, (
            f"{config_path.name}: DTYPE '{config['DTYPE']}' not in {valid_dtypes}"
        )

    @pytest.mark.parametrize("config_path", get_config_files(), ids=lambda p: p.name)
    def test_model_name_is_lowercase_alphanumeric(self, config_path):
        config = parse_env_file(config_path)
        name = config["MODEL_NAME"]
        # Допускаем буквы, цифры, дефисы, подчёркивания, точки
        assert all(
            c.isalnum() or c in "-_." for c in name
        ), f"{config_path.name}: MODEL_NAME '{name}' contains invalid characters"

    @pytest.mark.parametrize("config_path", get_config_files(), ids=lambda p: p.name)
    def test_extra_args_contains_trust_remote_code(self, config_path):
        config = parse_env_file(config_path)
        extra = config.get("EXTRA_ARGS", "")
        assert "--trust-remote-code" in extra, (
            f"{config_path.name}: EXTRA_ARGS should contain --trust-remote-code"
        )
