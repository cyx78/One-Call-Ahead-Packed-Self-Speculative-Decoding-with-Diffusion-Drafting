import pytest
from nemotron_ltr.scoring import _last_boxed, _score_output, score_output
from nemotron_ltr import evalplus_mbpp_base


def test_math_box_extraction_preserves_nested_braces():
    text = r"First \boxed{1}, finally \boxed{\frac{3}{4}}."
    assert _last_boxed(text) == r"\frac{3}{4}"
    assert _score_output("math500", text, {"answer": r"\frac{3}{4}"})["correct"]


def test_numeric_gsm_answer_extraction_and_missing_answer():
    assert score_output("3B", "gsm8k", r"\boxed{1,250}", {"answer": "#### 1250"})[
        "correct"
    ]
    assert score_output("8B", "gsm8k", r"\boxed{1250}", {"answer": "#### 1250"})[
        "correct"
    ]
    assert not score_output("3B", "gsm8k", "No result.", {"answer": "#### 1250"})[
        "correct"
    ]


def test_mbpp_release_is_checked_before_oracle_execution(monkeypatch):
    import evalplus.data
    import evalplus.evaluate

    monkeypatch.setattr(evalplus_mbpp_base, "_ORACLE_CACHE", {})
    monkeypatch.setattr(evalplus.data, "get_mbpp_plus", lambda **kwargs: {})
    monkeypatch.setattr(evalplus.data, "get_mbpp_plus_hash", lambda **kwargs: "wrong")
    monkeypatch.setattr(
        evalplus.evaluate,
        "trusted_exec",
        lambda *args, **kwargs: pytest.fail("Oracle must not execute"),
    )
    with pytest.raises(ValueError, match="pinned"):
        evalplus_mbpp_base._base_oracle("v0.2.0", True)
