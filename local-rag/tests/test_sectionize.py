from backend.app.sectionize import build_section_hierarchy


def test_rejects_common_false_positive_lines():
    document = {
        "elements": [
            {"id": "1", "type": "text", "page": 1, "content": "arXiv:1903.12261v1 [cs.LG] 28 Mar 2019"},
            {"id": "2", "type": "text", "page": 1, "content": "1 INTRODUCTION", "font_size": 12, "bold": True},
            {"id": "3", "type": "text", "page": 1, "content": "T = 0", "font_size": 12, "bold": False},
            {"id": "4", "type": "text", "page": 1, "content": "5 X", "font_size": 12, "bold": False},
        ]
    }
    result = build_section_hierarchy(document)
    titles = [section["title"] for section in result["sections"]]
    assert titles == ["1 INTRODUCTION"]
