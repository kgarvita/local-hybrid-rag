"""Small diagnostic utility for inspecting extracted element candidates."""

import argparse

from .extractor import extract_pdf
from .sectionize import heading_candidate_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    args = parser.parse_args()

    document = extract_pdf(args.pdf)
    for element in document.get("elements", []):
        text = element.get("content", "")
        if not text:
            continue
        score, level = heading_candidate_score(element, element.get("font_size"))
        print(
            f"{element['id']} page={element['page']} type={element['type']} "
            f"score={score:.1f} level={level} text={text!r}"
        )


if __name__ == "__main__":
    main()
