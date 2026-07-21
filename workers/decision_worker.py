def make_decision(
    nlp_score: float,
    price_score: float,
    trust_score: float,
    thresholds: dict,
    review_score: float = 0.0
) -> str:
    """
    Evaluates listing scores against dynamic trust thresholds.

    Parameters
    ----------
    nlp_score    : Fraud probability from listing text analysis (0–1).
    price_score  : Pricing anomaly probability (0–1).
    trust_score  : Seller trust score (0–100).
    thresholds   : Dynamic thresholds derived from seller trust tier.
    review_score : RoBERTa fake-review probability (0–1). Defaults to 0
                   (neutral) when no reviews are provided.

    Returns
    -------
    'approved', 'suspended' (moderator review required), or 'rejected'.
    """
    nlp_thresh   = thresholds['nlp_threshold']
    price_thresh = thresholds['price_threshold']

    # Composite NLP signal: blend listing text fraud score with review
    # authenticity signal.  Review score raises the effective NLP signal
    # when reviews look fake, but cannot single-handedly reject a listing.
    effective_nlp = min((nlp_score * 0.75) + (review_score * 0.25), 1.0)

    # 1. High-confidence fraud: both text and price scream fraud → auto-reject
    if effective_nlp >= 0.85 and price_score >= 0.80:
        return 'rejected'

    # 2. Extreme deceptive text regardless of price → auto-reject
    if effective_nlp >= 0.88:
        return 'rejected'

    # 3. High-risk seller with blatantly suspicious reviews + high price anomaly
    if trust_score < 30 and review_score >= 0.75 and price_score >= 0.70:
        return 'rejected'

    # 4. High-risk seller with extreme price anomaly → auto-reject
    if trust_score < 30 and price_score >= 0.85:
        return 'rejected'

    # 5. Both composite NLP and price breach seller's dynamic threshold
    if effective_nlp >= nlp_thresh and price_score >= price_thresh:
        return 'suspended'

    # 6. Either composite NLP or price breaches threshold → suspend for review
    if effective_nlp >= nlp_thresh or price_score >= price_thresh:
        return 'suspended'

    # 7. Reviews look heavily manipulated even if listing text seems clean
    if review_score >= 0.70:
        return 'suspended'

    # 8. Unverified/new seller submitting high-value item → safety check
    if trust_score < 35 and price_score > 0.50:
        return 'suspended'

    # 9. All checks passed → auto-approve
    return 'approved'
