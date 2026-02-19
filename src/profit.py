

def compute_profit(tn, fn, loan_profit, lgd):
    return tn * loan_profit - fn * lgd


def profit_per_approved(total_profit, tn, fn):
    approved = tn + fn
    return total_profit / approved if approved > 0 else 0