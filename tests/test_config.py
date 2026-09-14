import os

import app.config as config_module


def test_load_environment_reads_dotenv_file(tmp_path, monkeypatch):
    key_name = "CO_SCIENTIST_TEST_DOTENV_KEY"
    monkeypatch.delenv(key_name, raising=False)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(f"{key_name}=loaded-from-dotenv\n", encoding="utf-8")

    config_module.load_environment(dotenv_path)

    assert os.environ[key_name] == "loaded-from-dotenv"