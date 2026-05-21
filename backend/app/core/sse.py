import json


def step_line(name: str, status: str, label: str, detail: str = "") -> str:
    return f"data: {json.dumps({'type': 'step', 'name': name, 'status': status, 'label': label, 'detail': detail}, ensure_ascii=False)}\n\n"
