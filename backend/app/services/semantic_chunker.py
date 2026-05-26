from app.services.document_parser import ParsedDocument


def chunk_document(parsed: ParsedDocument, max_chunk_chars: int = 800) -> list[dict]:
    chunks = []
    current_path = []
    chunk_index = 0

    for section in parsed.sections:
        level = section.get("level", 0)
        title = section.get("title", "")
        content = section.get("content", "")

        # Update title path stack
        while len(current_path) >= level and level > 0:
            current_path.pop()
        if title:
            current_path.append(title)

        title_path = " > ".join(current_path) if current_path else title

        if not content:
            continue

        if len(content) <= max_chunk_chars:
            chunks.append({
                "content": content,
                "title_path": title_path,
                "chunk_index": chunk_index,
                "level": level,
            })
            chunk_index += 1
        else:
            # Split by paragraphs, avoid cutting sentences
            paragraphs = content.split("\n")
            current_chunk = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if current_chunk and len(current_chunk) + len(para) + 1 > max_chunk_chars:
                    chunks.append({
                        "content": current_chunk,
                        "title_path": title_path,
                        "chunk_index": chunk_index,
                        "level": level,
                    })
                    chunk_index += 1
                    current_chunk = para
                else:
                    current_chunk = f"{current_chunk}\n{para}".strip() if current_chunk else para
            if current_chunk:
                chunks.append({
                    "content": current_chunk,
                    "title_path": title_path,
                    "chunk_index": chunk_index,
                    "level": level,
                })
                chunk_index += 1

    return chunks
