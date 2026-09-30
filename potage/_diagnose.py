"""R^2 curve diagnosis.

diagnose_curve categorizes the shape of an R^2 selection curve into
actionable categories (steep-to-1, slow-climb, hard-plateau, moderate-plateau).
"""



def diagnose_curve(history, threshold_steep=0.95, threshold_moderate=0.80):
    """Diagnose R^2 curve shape.

    Categories:
        steep-to-1       : reaches > threshold_steep in <= 3 steps
        slow-climb       : monotonic but never reaches threshold_moderate
        hard-plateau     : reaches a max then stalls (max < 0.90)
        moderate-plateau : climbs to 0.80-0.95 then diminishing returns
        empty            : no selection steps

    Parameters
    ----------
    history : SelectionHistory
    threshold_steep : float
        R^2 threshold for "steep-to-1" category.
    threshold_moderate : float
        R^2 threshold separating slow-climb from moderate-plateau.

    Returns
    -------
    str : one of the category names above.
    """
    if history.score_kind != "r2":
        raise ValueError("diagnose_curve requires an R² selection history")
    if len(history) == 0:
        return 'empty'

    r2_vals = history.cumulative_r2()
    final = r2_vals[-1]

    for i, r2 in enumerate(r2_vals):
        if r2 >= threshold_steep and i < 3:
            return 'steep-to-1'

    if final < 0.50:
        return 'slow-climb'

    if len(r2_vals) >= 4:
        last_gain = r2_vals[-1] - r2_vals[-4]
        if last_gain < 0.01:
            if final < 0.90:
                return 'hard-plateau'
            return 'moderate-plateau'

    if final < threshold_moderate:
        return 'slow-climb'

    return 'moderate-plateau'
