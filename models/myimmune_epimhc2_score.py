"""
myimmune_epimhc2_score.py
-------------------------
Hosted API client for MyImmune EpiMHC2 v10.0.0.
Queries the private hosted MyImmune EpiMHC2 inference service via REST API.
No model weights, neural architectures, or proprietary assets are stored locally.
"""

import os
import requests

# No default endpoint is published. The hosted EpiMHC2 service is access-controlled;
# set MYIMMUNE_EPIMHC2_URL and MYIMMUNE_API_KEY to the credentials issued to you.
# See models/myimmune_epimhc2_README.md for how to request access.
ENDPOINT_URL = os.environ.get("MYIMMUNE_EPIMHC2_URL", "")
API_KEY = os.environ.get("MYIMMUNE_API_KEY", "")


def epimhc2_score(heavy_seq: str, light_seq: str = "", mode: str = "seq", url: str = "") -> float:
    """Score therapeutic antibody immunogenicity risk via MyImmune EpiMHC2 v10.0.0 hosted API.

    Parameters
    ----------
    heavy_seq : str
        Heavy chain variable domain sequence (VH).
    light_seq : str, optional
        Light chain variable domain sequence (VL).
    mode : str, optional
        seq for sequence-intrinsic IEDB-27 x OAS risk (default).
        clin for clinical pharmacology-adjusted ADA risk.
    url : str, optional
        REST API endpoint URL. Defaults to $MYIMMUNE_EPIMHC2_URL.

    Returns
    -------
    float
        Predicted immunogenicity risk score.
    """
    url = url or ENDPOINT_URL
    if not url:
        raise RuntimeError(
            "No EpiMHC2 endpoint configured. Set MYIMMUNE_EPIMHC2_URL (and MYIMMUNE_API_KEY) "
            "to the hosted endpoint issued to you; see models/myimmune_epimhc2_README.md."
        )

    payload = {
        "constructs": [{"vh": heavy_seq.strip(), "vl": light_seq.strip()}],
        "window": 15
    }
    headers = {
        "User-Agent": "MyImmune-FLAb-Client/1.0",
        "Content-Type": "application/json"
    }
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    res = resp.json()["results"][0]
    return float(res["v10_clin_score"] if mode == "clin" else res["v10_seq_score"])
