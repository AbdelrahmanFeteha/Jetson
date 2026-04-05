import requests

def get_command(api_base: str, timeout: float = 3.0) -> dict:
    r = requests.get(f"{api_base}/command", timeout=timeout)
    r.raise_for_status()
    return r.json()

def post_inspection(api_base: str, payload: dict, timeout: float = 10.0) -> dict:
    r = requests.post(f"{api_base}/inspection", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def post_telemetry(api_base: str, payload: dict, timeout: float = 3.0) -> dict:
    r = requests.post(f"{api_base}/telemetry", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()