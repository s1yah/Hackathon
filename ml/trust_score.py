class SellerTrustScorer:
    """
    Computes a seller trust score between 0 (untrusted) and 100 (fully trusted)
    and provides dynamic anomaly detection thresholds based on trust level.
    """
    WEIGHTS = {
        'account_age': 0.20,
        'verification_status': 0.30,
        'successful_sales_rate': 0.25,
        'flag_rate': 0.25
    }

    def calculate(self, seller_data: dict) -> float:
        age_days = seller_data.get('account_age_days', 0)
        verification = seller_data.get('verification_status', 'unverified').lower()
        total_listings = seller_data.get('total_listings', 0)
        flagged = seller_data.get('flagged_listings', 0)
        sales = seller_data.get('successful_sales', 0)

        # Account age score (maxed out at 365 days)
        age_score = min(age_days / 365.0, 1.0) * 100

        # Verification score
        verification_score = {
            'verified': 100,
            'unverified': 40,
            'new': 10
        }.get(verification, 20)

        # Sales success rate
        sales_rate = (sales / max(total_listings, 1)) if total_listings > 0 else (1.0 if verification == 'verified' else 0.5)
        sales_score = min(sales_rate, 1.0) * 100

        # Flag rate penalty
        flag_rate = (flagged / max(total_listings, 1)) if total_listings > 0 else 0.0
        flag_score = max(1.0 - flag_rate, 0.0) * 100

        # Calculate weighted average
        trust_score = (
            age_score * self.WEIGHTS['account_age'] +
            verification_score * self.WEIGHTS['verification_status'] +
            sales_score * self.WEIGHTS['successful_sales_rate'] +
            flag_score * self.WEIGHTS['flag_rate']
        )

        return round(min(max(trust_score, 0.0), 100.0), 2)

    def get_thresholds(self, trust_score: float) -> dict:
        """
        Returns dynamic thresholds for NLP & Pricing models.
        High-trust sellers -> higher tolerance (more lenient).
        Low-trust/new sellers -> lower tolerance (stricter scrutiny).
        """
        if trust_score >= 75:
            return {'nlp_threshold': 0.75, 'price_threshold': 0.85, 'tier': 'Trusted'}
        elif trust_score >= 40:
            return {'nlp_threshold': 0.60, 'price_threshold': 0.70, 'tier': 'Standard'}
        else:
            return {'nlp_threshold': 0.45, 'price_threshold': 0.55, 'tier': 'High Risk / New'}
