"""Exercise installed dependency guards with owned fixtures and no remote code."""

from unittest.mock import Mock

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.generation import utils as generation
from transformers.utils import CHAT_TEMPLATE_DIR


@pytest.mark.parametrize("local", [False, True])
def test_declined_custom_generation_does_not_fetch_code(monkeypatch, tmp_path, local):
    source = str(tmp_path) if local else "owned-fixture/declined-generation"
    monkeypatch.setattr(generation, "has_file", lambda *a, **kw: True, raising=False)
    fetch = Mock(side_effect=AssertionError("Code fetched before consent"))
    monkeypatch.setattr(generation, "get_cached_module_file", fetch)
    with pytest.raises(ValueError, match="trust_remote_code"):
        generation.GenerationMixin.load_custom_generate(
            object(), source, trust_remote_code=False
        )
    fetch.assert_not_called()


def test_explicitly_trusted_custom_generation_keeps_supported_loading(monkeypatch):
    events = []
    monkeypatch.setattr(generation, "has_file", lambda *a, **kw: True, raising=False)

    def fetch(*args, **kwargs):
        events.append("fetch")
        return "owned-fixture-module"

    def requirements(*args, **kwargs):
        events.append("requirements")

    def owned_generate():
        return "owned result"

    def load(name, module):
        assert (name, module) == ("generate", "owned-fixture-module")
        events.append("load")
        return owned_generate

    monkeypatch.setattr(generation, "get_cached_module_file", fetch)
    monkeypatch.setattr(generation, "check_python_requirements", requirements)
    monkeypatch.setattr(generation, "get_class_in_module", load)
    loaded = generation.GenerationMixin.load_custom_generate(
        object(), "owned-fixture/trusted-generation", trust_remote_code=True
    )
    assert loaded() == "owned result"
    assert events == ["fetch", "requirements", "load"]


def fixture_tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )


@pytest.mark.parametrize("name", ["../outside", "nested/template", "absolute"])
def test_chat_template_names_cannot_escape_export(tmp_path, name):
    tokenizer = fixture_tokenizer()
    export = tmp_path / "export"
    forbidden = tmp_path / "outside.jinja"
    forbidden.write_text("owned sentinel")
    if name == "absolute":
        name = str(forbidden.with_suffix(""))
    tokenizer.chat_template = {name: "owned replacement"}
    with pytest.raises(ValueError, match="Invalid chat template name"):
        tokenizer.save_pretrained(export)
    assert forbidden.read_text() == "owned sentinel"
    assert sorted(
        p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.jinja")
    ) == ["outside.jinja"]


def test_safe_named_and_default_chat_templates_still_roundtrip(tmp_path):
    tokenizer = fixture_tokenizer()
    templates = {"default": "{{ messages }}", "owned": "Owned {{ messages }}"}
    tokenizer.chat_template = templates
    export = tmp_path / "export"
    tokenizer.save_pretrained(export)
    assert (export / "chat_template.jinja").read_text() == templates["default"]
    assert (export / CHAT_TEMPLATE_DIR / "owned.jinja").read_text() == templates[
        "owned"
    ]
    restored = PreTrainedTokenizerFast.from_pretrained(export, local_files_only=True)
    assert restored.chat_template == templates
