"""Targeted compatibility fix for public Zorua Illusion reveals."""

import hashlib
import inspect
import textwrap


def install_illusion_family():
    from metamon.backend.replay_parser.forward import SimProtocol

    handler = SimProtocol._parse_replace
    if getattr(handler, "_fog_illusion_family", False):
        return "zoroark-zorua-replace-v1"
    source = inspect.getsource(handler)
    expected = "67df1b1e8970235ea1fa2718549fa884ad5981817729b54c15ee80696cee941a"
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise ValueError("Metamon replace handler changed; audit Illusion compatibility")
    original = 'replace_with.name.startswith("Zoroark")'
    if source.count(original) != 1:
        raise ValueError("Unexpected Illusion family guard")
    source = source.replace(original, 'replace_with.name.startswith(("Zoroark", "Zorua"))')
    namespace = {}
    exec(
        compile(textwrap.dedent(source), handler.__code__.co_filename, "exec"),
        handler.__globals__,
        namespace,
    )
    SimProtocol._parse_replace = namespace[handler.__name__]
    SimProtocol._parse_replace._fog_illusion_family = True
    return "zoroark-zorua-replace-v1"
