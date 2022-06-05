class Filter:
    """
    Allow all filters to have this properties, even if none.
    """
    crop = None
    size = None
    cords = None


class CROP_SPEEDLIMIT_SCREEN(Filter):
    """
    Speed-limit crop
    -> generates smaller inputs for search-areas crops
    """
    crop = (0, 600, 400, 800)


class CROP_SPEEDLIMIT_AREA(Filter):
    """
    Speed-limit search area
    -> generates smaller inputs for detection
    """
    size = (200, 200)
    cords = (
            (0, 0), (100, 0), (200, 0), (300, 0), (400, 0),
            (0, 100), (100, 100), (200, 100), (300, 100), (400, 100),
            (0, 200), (100, 200), (200, 200), (300, 200), (400, 200)
        )