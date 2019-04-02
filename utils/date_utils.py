import datetime
from warnings import warn

def get_surgery_date(house_id):
    """Return the surgery date for some available houses"""
    if house_id == 4954:
        surgery_date = datetime.datetime(2017, 10, 21)
    elif house_id == 8145:
        surgery_date = datetime.datetime(2018, 3, 21)
    elif house_id == 3590:
        surgery_date = datetime.datetime(2018, 1, 12)
    elif house_id == 8731:
        surgery_date = datetime.datetime(2018, 2, 22)
    else:
        warn('Surgery date not available for {}'.format(house_id))
        surgery_date = None
    return surgery_date

def parse_time(x):
    """Parse the time the way it was saved in the SPHERE database"""
    try:
        y = datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S.%f').timestamp()
    except ValueError:
        y = datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S').timestamp()
    return y