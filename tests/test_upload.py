"""Auto-upload of a run's outputs to the Databricks volume (run.upload_run_outputs), with the
Databricks calls mocked: no network."""
from bessrank import config, databricks, run


def _setup(tmp_path, monkeypatch):
    forecasts = tmp_path / "test_forecasts.parquet"
    forecasts.write_bytes(b"x" * 1000)
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}")
    monkeypatch.setattr(config, "PROVENANCE_JSON", provenance)
    return forecasts, provenance


def test_upload_succeeds_into_a_folder_named_by_run_type_and_commit(tmp_path, monkeypatch):
    forecasts, provenance = _setup(tmp_path, monkeypatch)
    sizes = {}

    def fake_upload(path, remote):
        sizes[remote] = path.stat().st_size
        return True
    monkeypatch.setattr(databricks, "upload_file", fake_upload)
    monkeypatch.setattr(databricks, "remote_file_size", lambda remote: sizes.get(remote))
    assert run.upload_run_outputs("test", [forecasts], commit="0123456789abcdef0123")
    assert set(sizes) == {"runs/test-0123456789ab/test_forecasts.parquet", "runs/test-0123456789ab/provenance.json"}


def test_missing_forecast_file_is_skipped_and_provenance_still_uploaded(tmp_path, monkeypatch):
    _, provenance = _setup(tmp_path, monkeypatch)
    uploaded = []
    monkeypatch.setattr(databricks, "upload_file", lambda path, remote: uploaded.append(remote) or True)
    monkeypatch.setattr(databricks, "remote_file_size", lambda remote: provenance.stat().st_size)
    assert run.upload_run_outputs("explore-analyses", [tmp_path / "absent.parquet"], commit="abc-dirty")
    assert uploaded == ["runs/explore-analyses-abc-dirty/provenance.json"]


def test_failed_upload_warns_keeps_local_files_and_does_not_raise(tmp_path, monkeypatch, capsys):
    forecasts, provenance = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(databricks, "upload_file", lambda path, remote: False)
    monkeypatch.setattr(databricks, "remote_file_size", lambda remote: None)
    assert not run.upload_run_outputs("test", [forecasts], commit="0" * 40)
    assert "WARNING" in capsys.readouterr().out
    assert forecasts.exists() and provenance.exists()


def test_size_mismatch_counts_as_a_failure(tmp_path, monkeypatch, capsys):
    forecasts, _ = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(databricks, "upload_file", lambda path, remote: True)
    monkeypatch.setattr(databricks, "remote_file_size", lambda remote: 1)  # truncated upload
    assert not run.upload_run_outputs("test", [forecasts], commit="0" * 40)
    assert "test_forecasts.parquet" in capsys.readouterr().out


def test_network_error_is_caught(tmp_path, monkeypatch, capsys):
    forecasts, _ = _setup(tmp_path, monkeypatch)

    def boom(path, remote):
        raise RuntimeError("DATABRICKS_HOST is not set")
    monkeypatch.setattr(databricks, "upload_file", boom)
    assert not run.upload_run_outputs("test", [forecasts], commit="0" * 40)
    assert "WARNING" in capsys.readouterr().out
