from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import requests

from fog.scripts.kaggle_results import download_output


def response(status, headers, chunks):
    return nullcontext(SimpleNamespace(status_code=status, headers=headers,
                                      iter_content=lambda size: iter(chunks)))


def test_interrupted_transfer_resumes_suffix_and_publishes_atomically(tmp_path, monkeypatch):
    output = tmp_path / "result.zip"
    output.write_bytes(b"previous valid output")
    calls = []

    def interrupted():
        yield b"first"
        assert output.read_bytes() == b"previous valid output"
        raise requests.exceptions.ChunkedEncodingError("interrupted signed-url-with-secret")

    def get(url, **kwargs):
        calls.append(kwargs["headers"])
        if len(calls) == 1:
            return nullcontext(SimpleNamespace(status_code=200,
                headers={"Content-Length": "11", "ETag": '"stable"'},
                iter_content=lambda size: interrupted()))
        assert (tmp_path / "result.zip.part").read_bytes() == b"first"
        return response(206, {"Content-Length": "6", "Content-Range": "bytes 5-10/11",
                              "ETag": '"stable"'}, [b"suffix"])

    monkeypatch.setattr(requests, "get", get)
    download_output("https://example.invalid/private", output)
    assert calls == [{"Accept-Encoding": "identity"},
                     {"Accept-Encoding": "identity", "Range": "bytes=5-", "If-Range": '"stable"'}]
    assert output.read_bytes() == b"firstsuffix"
    assert not (tmp_path / "result.zip.part").exists()


def test_ignored_range_restarts_body_without_duplicate_prefix(tmp_path, monkeypatch):
    calls = []

    def interrupted():
        yield b"old"
        raise requests.ConnectionError("network")

    def get(url, **kwargs):
        calls.append(kwargs["headers"])
        if len(calls) == 1:
            return nullcontext(SimpleNamespace(status_code=200,
                headers={"Content-Length": "6", "ETag": '"old"'},
                iter_content=lambda size: interrupted()))
        return response(200, {"Content-Length": "8", "ETag": '"new"'}, [b"complete"])

    monkeypatch.setattr(requests, "get", get)
    output = tmp_path / "result.zip"
    download_output("https://example.invalid/private", output)
    assert calls[1]["Range"] == "bytes=3-"
    assert output.read_bytes() == b"complete"


@pytest.mark.parametrize("headers", [
    {"Content-Range": "bytes 0-5/9", "ETag": '"stable"'},
    {"Content-Range": "bytes 3-8/10", "ETag": '"stable"'},
    {"Content-Range": "bytes 3-8/9", "ETag": '"changed"'},
    {"Content-Range": "bytes 3-8/9", "Content-Length": "7", "ETag": '"stable"'},
])
def test_resume_rejects_mismatched_object_or_range(tmp_path, monkeypatch, headers):
    calls = []

    def interrupted():
        yield b"abc"
        raise requests.ConnectionError("network")

    def get(url, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return nullcontext(SimpleNamespace(status_code=200,
                headers={"Content-Length": "9", "ETag": '"stable"'},
                iter_content=lambda size: interrupted()))
        return response(206, headers, [b"defghi"])

    monkeypatch.setattr(requests, "get", get)
    output = tmp_path / "result.zip"
    with pytest.raises(RuntimeError, match="range|length|changed"):
        download_output("https://example.invalid/private", output)
    assert not output.exists()
    assert (tmp_path / "result.zip.part").read_bytes() == b"abc"


def test_bounded_failure_keeps_existing_output_and_hides_signed_url(tmp_path, monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs)
        raise requests.ConnectionError("private-signed-url-with-secret")

    monkeypatch.setattr(requests, "get", get)
    output = tmp_path / "result.zip"
    output.write_bytes(b"existing")
    with pytest.raises(RuntimeError, match="after 3 attempts") as error:
        download_output("https://example.invalid/private", output)
    assert len(calls) == 3 and "secret" not in str(error.value)
    assert output.read_bytes() == b"existing"


def test_clean_but_short_response_is_not_published(tmp_path, monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs)
        return response(200, {"Content-Length": "10"}, [b"short"])

    monkeypatch.setattr(requests, "get", get)
    output = tmp_path / "result.zip"
    with pytest.raises(RuntimeError, match="incomplete"):
        download_output("https://example.invalid/private", output)
    assert len(calls) == 3 and not output.exists()
