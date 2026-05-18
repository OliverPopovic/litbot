import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True)
class CorpusWork:
    source_id: str
    filename: str
    title: str
    author: str
    publication_year: int
    genre: str
    uri: str
    url: str
    work: str
    translator: str | None = None


DEFAULT_PUBLIC_DOMAIN_WORKS: tuple[CorpusWork, ...] = (
    CorpusWork(
        source_id="pride-prejudice-1813",
        filename="pride-prejudice-1813.txt",
        title="Pride and Prejudice",
        author="Jane Austen",
        publication_year=1813,
        genre="novel",
        uri="https://www.gutenberg.org/ebooks/1342",
        url="https://www.gutenberg.org/ebooks/1342.txt.utf-8",
        work="Pride and Prejudice",
    ),
    CorpusWork(
        source_id="moby-dick-1851",
        filename="moby-dick-1851.txt",
        title="Moby Dick; Or, The Whale",
        author="Herman Melville",
        publication_year=1851,
        genre="novel",
        uri="https://www.gutenberg.org/ebooks/2701",
        url="https://www.gutenberg.org/ebooks/2701.txt.utf-8",
        work="Moby-Dick",
    ),
    CorpusWork(
        source_id="hamlet",
        filename="hamlet.txt",
        title="Hamlet",
        author="William Shakespeare",
        publication_year=1603,
        genre="play",
        uri="https://www.gutenberg.org/ebooks/1524",
        url="https://www.gutenberg.org/ebooks/1524.txt.utf-8",
        work="Hamlet",
    ),
    CorpusWork(
        source_id="father-goriot-1835",
        filename="father-goriot-1835.txt",
        title="Father Goriot",
        author="Honoré de Balzac",
        translator="Ellen Marriage",
        publication_year=1835,
        genre="novel",
        uri="https://www.gutenberg.org/ebooks/1237",
        url="https://www.gutenberg.org/ebooks/1237.txt.utf-8",
        work="Father Goriot",
    ),
    CorpusWork(
        source_id="crime-punishment-1866",
        filename="crime-punishment-1866.txt",
        title="Crime and Punishment",
        author="Fyodor Dostoyevsky",
        translator="Constance Garnett",
        publication_year=1866,
        genre="novel",
        uri="https://www.gutenberg.org/ebooks/2554",
        url="https://www.gutenberg.org/ebooks/2554.txt.utf-8",
        work="Crime and Punishment",
    ),
    CorpusWork(
        source_id="don-quixote-1605",
        filename="don-quixote-1605.txt",
        title="Don Quixote",
        author="Miguel de Cervantes Saavedra",
        translator="John Ormsby",
        publication_year=1605,
        genre="novel",
        uri="https://www.gutenberg.org/ebooks/996",
        url="https://www.gutenberg.org/ebooks/996.txt.utf-8",
        work="Don Quixote",
    ),
    CorpusWork(
        source_id="odyssey",
        filename="odyssey.txt",
        title="The Odyssey",
        author="Homer",
        translator="Alexander Pope",
        publication_year=-750,
        genre="poetry",
        uri="https://www.gutenberg.org/ebooks/3160",
        url="https://www.gutenberg.org/ebooks/3160.txt.utf-8",
        work="The Odyssey",
    ),
    CorpusWork(
        source_id="madame-bovary-1857",
        filename="madame-bovary-1857.txt",
        title="Madame Bovary",
        author="Gustave Flaubert",
        translator="Eleanor Marx Aveling",
        publication_year=1857,
        genre="novel",
        uri="https://www.gutenberg.org/ebooks/2413",
        url="https://www.gutenberg.org/ebooks/2413.txt.utf-8",
        work="Madame Bovary",
    ),
)

START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
FRONT_MATTER_RE = re.compile(
    r"\b(contents|illustrations|list of illustrations|preface|introduction|etymology|extracts)\b",
    re.I,
)
TEXT_START_RE = re.compile(
    r"(?m)^\s*(chapter|act|book)\s+(?:[ivxlcdm]+|\d+)\b[^\n]*$",
    re.I,
)


def fetch_public_domain_corpus(
    target: Path,
    works: tuple[CorpusWork, ...] = DEFAULT_PUBLIC_DOMAIN_WORKS,
) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for work in works:
        raw_text = _download_text(work.url)
        text = strip_gutenberg_boilerplate(raw_text)
        text = strip_front_matter(text)
        text_path = target / work.filename
        metadata_path = text_path.with_suffix(text_path.suffix + ".json")
        text_path.write_text(text, encoding="utf-8")
        metadata_path.write_text(
            json.dumps(metadata_for_work(work), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.extend([text_path, metadata_path])
    return written


def strip_gutenberg_boilerplate(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    start = 0
    for index, line in enumerate(lines):
        if START_RE.search(line):
            start = index + 1
            break

    end = len(lines)
    for index in range(start, len(lines)):
        if END_RE.search(lines[index]):
            end = index
            break

    return "\n".join(lines[start:end]).strip() + "\n"


def strip_front_matter(text: str) -> str:
    """Remove obvious non-literary front matter before an early chapter/act/book marker."""

    search_window = text[:30000]
    for match in TEXT_START_RE.finditer(search_window):
        preceding = search_window[: match.start()]
        if not FRONT_MATTER_RE.search(preceding):
            continue
        if _next_nonblank_line_is_text_start(search_window, match.end()):
            continue
        return text[match.start() :].lstrip()
    return text


def _next_nonblank_line_is_text_start(text: str, start: int) -> bool:
    for line in text[start:].splitlines():
        if not line.strip():
            continue
        return TEXT_START_RE.match(line) is not None
    return False


def metadata_for_work(work: CorpusWork) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source_id": work.source_id,
        "title": work.title,
        "author": work.author,
        "publication_year": work.publication_year,
        "genre": work.genre,
        "language": "en",
        "license": "Public domain",
        "uri": work.uri,
        "version": "1",
        "metadata": {"work": work.work},
    }
    if work.translator:
        metadata["translator"] = work.translator
    return metadata


def _download_text(url: str) -> str:
    with urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")
