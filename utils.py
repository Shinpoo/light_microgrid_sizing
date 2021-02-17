from datetime import date

def add_years(d, years):
    """
    Return a date that's `years` years after the date (or datetime)
    object `d`. Return the same calendar date (month and day) in the
    destination year, if it exists, otherwise use the following day
    (thus changing February 29 to March 1).

    :param d: date or datetime object
    :param years: int number of years
    :return: a new object of same type as `date` with year += years
    """
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d + (date(d.year + years, 1, 1) - date(d.year, 1, 1))