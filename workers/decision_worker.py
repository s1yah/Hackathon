def make_decision(nlp_score: float, price_score: float, trust_score: float, thresholds: dict) -> str:
    """
    Evaluates listing scores against dynamic trust thresholds.
    Returns: 'approved', 'suspended' (moderator review required), or 'rejected'.
    """
    nlp_thresh = thresholds['nlp_threshold']
    price_thresh = thresholds['price_threshold']

    # 1. High confidence fraud (both NLP and price flag high risk) -> auto-reject
    if nlp_score >= 0.85 and price_score >= 0.80:
        return 'rejected'

    # 2. Extreme deceptive text score regardless of price -> auto-reject
    if nlp_score >= 0.88:
        return 'rejected'

    # 3. High risk seller with high price anomaly -> auto-reject
    if trust_score < 30 and price_score >= 0.85:
        return 'rejected'

    # 4. Both signals breach seller's dynamic threshold -> suspend for human review
    if nlp_score >= nlp_thresh and price_score >= price_thresh:
        return 'suspended'

    # 5. Either signal breaches threshold -> suspend for human review
    if nlp_score >= nlp_thresh or price_score >= price_thresh:
        return 'suspended'

    # 6. Unverified/new seller submitting high-value item -> suspend for safety verification
    if trust_score < 35 and price_score > 0.50:
        return 'suspended'

    # 7. Passes all checks cleanly -> auto-approve
    return 'approved'
