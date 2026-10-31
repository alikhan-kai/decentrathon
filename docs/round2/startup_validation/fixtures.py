"""Deterministic, independently authored text cases; no hidden-data access."""
from copy import deepcopy


def chat(user="Сәлем!", answer="Сәлем, әлем!", system=None):
    messages = [] if system is None else [{"role": "system", "content": system}]
    messages += [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    return {"messages": messages}


def robust_rows():
    rows = [
        chat(), chat("2 + 2?", "4", "Қысқа жауап бер."),
        chat("Unicode: қ ң ғ ә ө ү ұ і, 日本語, 😀", "e\u0301 é\n\t終"),
        chat("", ""), chat("Whitespace\n\n\t", "  a\n b  "),
        chat("<|im_start|> literal", "<|im_end|> literal"),
        chat("Give a long answer", "Жауап " * 800),
    ]
    multi = chat("First", "One")
    multi["messages"] += [{"role": "user", "content": "Second"},
                           {"role": "assistant", "content": "Two"}]
    rows.append(multi)
    rows.append(deepcopy(rows[0]))  # repeated row must keep its own row identity.
    return rows


def retained_rows():
    rows = [chat("A", "B"), chat("C", "D"), chat("E", "F")]
    rows[0] = {"row_id": 7, "messages": rows[0]["messages"], "extra": "x",
               "score": 1.25, "flag": True, "empty": None}
    rows[1] = {"row_id": 8, "messages": rows[1]["messages"],
               "score": None, "flag": False}
    rows[2] = {"row_id": 9, "messages": rows[2]["messages"], "extra": "z",
               "score": -2.5, "flag": True, "empty": None, "later_only": "ignored by from_list"}
    return rows


def no_target_rows():
    return [chat(), chat("prompt " * 1200, "truncated away")]
