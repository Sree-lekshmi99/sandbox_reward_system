from io import BytesIO
from pathlib import Path
from urllib.error import URLError
import json
import pytest
import bedside.ollama_backend as module
from bedside.dataset import load_cases, read_jsonl
from bedside.ollama_backend import OllamaClient, collect_bank, load_bank, message_hash


def test_ollama_uses_nonstreaming_json_and_preserves_output(monkeypatch):
    def mock_urlopen(request, timeout):
        assert request.full_url == "http://localhost:11434/api/chat"
        payload = json.loads(request.data)
        assert payload["stream"] is False
        assert payload["model"] == "installed-test-model"
        assert payload["options"]["seed"] == 19
        assert payload["messages"][-1]["content"] == "fictional message"
        return BytesIO(json.dumps({"message": {"content": "  Untouched reply.\n"}}).encode())
    monkeypatch.setattr(module, "urlopen", mock_urlopen)
    assert OllamaClient("installed-test-model").generate("fictional message", 19)["response"] == "  Untouched reply.\n"


def test_ollama_network_failure_is_actionable(monkeypatch):
    def fail(*args, **kwargs):
        raise URLError("offline")
    monkeypatch.setattr(module, "urlopen", fail)
    with pytest.raises(RuntimeError, match="Start the local server"):
        OllamaClient("model").generate("fictional message", 7)


def test_ollama_rejects_empty_content(monkeypatch):
    monkeypatch.setattr(module, "urlopen", lambda *a, **k: BytesIO(b'{"message":{"content":""}}'))
    with pytest.raises(ValueError, match="empty"):
        OllamaClient("model").generate("fictional message", 7)


class FakeClient:
    model = "mock-model-not-a-real-run"
    base_url = "http://localhost:11434"
    calls = 0
    def generate(self, message, seed):
        self.calls += 1
        return {"response": "  A fake response preserved for an adapter test.\n",
                "reported_model": self.model, "done_reason": "stop", "eval_count": 10}


def test_collection_resume_and_bank_validation(tmp_path):
    client = FakeClient()
    path = tmp_path / "bank.jsonl"
    assert collect_bank(client, path, n=1) == 12
    assert client.calls == 12
    bank = load_bank(path)
    assert len(bank) == 12
    assert all(len(v) == 1 for v in bank.values())
    assert collect_bank(client, path, n=1, resume=True) == 12
    assert client.calls == 12
    with pytest.raises(FileExistsError):
        collect_bank(client, path, n=1)
    with pytest.raises(ValueError, match="configuration"):
        collect_bank(client, path, n=2, resume=True)


def test_bank_rejects_duplicate_candidate(tmp_path):
    c = load_cases("test")[0]
    row = dict(case_id=c.id, candidate_index=0, message_sha256=message_hash(c.message), response="Test.")
    path = tmp_path / "duplicate.jsonl"
    path.write_text((json.dumps(row) + '\n') * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_bank(path)


def test_bank_rejects_mismatched_patient_text(tmp_path):
    c = load_cases("test")[0]
    path = tmp_path / "wrong-hash.jsonl"
    path.write_text(json.dumps(dict(case_id=c.id, candidate_index=0, message_sha256="wrong", response="Test."))+'\n')
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bank(path)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://user:password@localhost", "ftp://localhost"])
def test_client_rejects_invalid_base_urls(url):
    with pytest.raises(ValueError):
        OllamaClient("model", base_url=url)
