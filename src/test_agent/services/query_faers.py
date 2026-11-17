"""Plain (non-agent) client for the openFDA FAERS drug adverse event endpoint."""

import httpx

FAERS_URL = "https://api.fda.gov/drug/event.json"
REACTION_FIELD = "patient.reaction.reactionmeddrapt.exact"
NAME_FIELDS = ("patient.drug.openfda.generic_name", "patient.drug.openfda.brand_name")


def _clean(name: str | None) -> str:
    # Quotes would break the openFDA phrase syntax, so drop them.
    return (name or "").replace('"', "").strip()


def build_search(
    drug: str | None = None,
    brand_name: str | None = None,
    suspect_only: bool = True,
) -> str:
    """Build the openFDA `search` expression.

    Every name given is matched against both generic_name and brand_name and
    the clauses are OR'd, so "Tylenol" and "acetaminophen" both resolve.
    """
    names = list(dict.fromkeys(n for n in (_clean(drug), _clean(brand_name)) if n))
    if not names:
        raise ValueError("provide drug and/or brand_name")

    clauses = [f'{field}:"{name}"' for name in names for field in NAME_FIELDS]
    search = "(" + " OR ".join(clauses) + ")"
    if suspect_only:
        # drugcharacterization 1 = suspect drug (2 = concomitant, 3 = interacting).
        search += " AND patient.drug.drugcharacterization:1"
    return search


def query_faers(
    drug: str | None = None,
    limit: int = 10,
    brand_name: str | None = None,
    suspect_only: bool = True,
    timeout: float = 10.0,
) -> list[dict]:
    """Return the top adverse reactions reported for a drug in FAERS.

    Calls:
        GET api.fda.gov/drug/event.json
            ?search=(<name matches>) AND patient.drug.drugcharacterization:1
            &count=patient.reaction.reactionmeddrapt.exact
            &limit=<limit>

    Args:
        drug: generic name (e.g. "acetaminophen"); a brand name also works.
        limit: number of reaction terms to return (1-1000).
        brand_name: optional brand name (e.g. "Tylenol"). If both `drug` and
            `brand_name` are given, reports matching either are counted.
        suspect_only: only count reports with a drug flagged as suspect.
            openFDA can't tie this to the matched drug entry, so a report
            where the matched drug is concomitant but another drug is suspect
            still counts.

    Returns a list like [{"term": "NAUSEA", "count": 12345}, ...], sorted by
    count descending. Returns [] if openFDA has no matching reports.
    """
    if not 1 <= limit <= 1000:  # openFDA caps count queries at 1000 terms
        raise ValueError("limit must be between 1 and 1000")

    params = {
        "search": build_search(drug, brand_name, suspect_only),
        "count": REACTION_FIELD,
        "limit": limit,
    }
    resp = httpx.get(FAERS_URL, params=params, timeout=timeout)

    # openFDA returns 404 with error code NOT_FOUND when nothing matches.
    if resp.status_code == 404:
        return []
    resp.raise_for_status()

    return resp.json().get("results", [])[:limit]


if __name__ == "__main__":
    print(query_faers("ibuprofen", limit=5))
