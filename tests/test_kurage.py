from io import StringIO

from kurage import (
    dump_conversation,
    dumps_conversation,
    load_conversation,
    loads_conversation,
)


def test_loads_conversation_1():
    result = loads_conversation("user: hello\n")
    assert result == [{"role": "user", "content": "hello\n"}]


def test_loads_conversation_2():
    result = loads_conversation("assistant: hi\n")
    assert result == [{"role": "assistant", "content": "hi\n"}]


def test_loads_conversation_3():
    string = "user: hello\nassistant: hi\n"
    assert loads_conversation(string) == [
        {"role": "user", "content": "hello\n"},
        {"role": "assistant", "content": "hi\n"},
    ]


def test_loads_conversation_4():
    string = "user:\n  line1\n  line2\n"
    assert loads_conversation(string) == [{"role": "user", "content": "line1\nline2\n"}]


def test_loads_conversation_5():
    string = "user:\n  line1\n\n  line2\n"
    assert loads_conversation(string) == [
        {"role": "user", "content": "line1\n\nline2\n"}
    ]


def test_loads_conversation_6():
    assert loads_conversation("") == []


def test_dumps_conversion_1():
    messages = [{"role": "user", "content": "hello"}]
    assert dumps_conversation(messages) == "user: hello\n"


def test_dumps_conversion_2():
    messages = [{"role": "user", "content": "hello\n\n"}]
    assert dumps_conversation(messages) == "user: hello\n"


def test_dumps_conversion_3():
    messages = [{"role": "user", "content": "line1\nline2"}]
    assert dumps_conversation(messages) == "user: \n  line1\n  line2\n"


def test_dumps_conversion_4():
    messages = [{"role": "user", "content": "line1\nline2\n"}]
    assert dumps_conversation(messages) == "user: \n  line1\n  line2\n"


def test_dumps_conversion_5():
    assert dumps_conversation([]) == ""


def test_dumps_conversion_6():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    assert dumps_conversation(messages) == "user: hello\nassistant: world\n"


def test_load_conversation_1():
    fp = StringIO("user: hello\n")
    assert load_conversation(fp) == [{"role": "user", "content": "hello\n"}]


def test_dump_conversation_1():
    messages = [{"role": "user", "content": "hello"}]
    fp = StringIO()
    dump_conversation(messages, fp)
    assert fp.getvalue() == "user: hello\n"


def test_load_dump_conversion_1():
    original = "user: hello\nassistant: world\n"
    in_fp = StringIO(original)
    out_fp = StringIO()
    dump_conversation(load_conversation(in_fp), out_fp)
    assert out_fp.getvalue() == original
