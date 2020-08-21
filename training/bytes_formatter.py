"""
Borrowed from
https://github.com/Stonesjtu/calmsize/blob/19d7a3ad86068440b138dd6473d75c42e6f566c2/calmsize/calmsize.py
"""
traditional = [
    (1024 ** 5, 'P'),
    (1024 ** 4, 'T'),
    (1024 ** 3, 'G'),
    (1024 ** 2, 'M'),
    (1024 ** 1, 'K'),
    (1024 ** 0, 'B'),
]

alternative = [
    (1024 ** 5, ' PB'),
    (1024 ** 4, ' TB'),
    (1024 ** 3, ' GB'),
    (1024 ** 2, ' MB'),
    (1024 ** 1, ' KB'),
    (1024 ** 0, (' byte', ' bytes')),
]

verbose = [
    (1024 ** 5, (' petabyte', ' petabytes')),
    (1024 ** 4, (' terabyte', ' terabytes')),
    (1024 ** 3, (' gigabyte', ' gigabytes')),
    (1024 ** 2, (' megabyte', ' megabytes')),
    (1024 ** 1, (' kilobyte', ' kilobytes')),
    (1024 ** 0, (' byte', ' bytes')),
]

iec = [
    (1024 ** 5, 'Pi'),
    (1024 ** 4, 'Ti'),
    (1024 ** 3, 'Gi'),
    (1024 ** 2, 'Mi'),
    (1024 ** 1, 'Ki'),
    (1024 ** 0, ''),
]

si = [
    (1000 ** 5, 'P'),
    (1000 ** 4, 'T'),
    (1000 ** 3, 'G'),
    (1000 ** 2, 'M'),
    (1000 ** 1, 'K'),
    (1000 ** 0, 'B'),
]


# noinspection PyUnboundLocalVariable
class ByteSize:
    def __init__(self, num_bytes, system=None):
        self.num_bytes = num_bytes
        self.system = system or traditional
        self.amount = num_bytes
        self.unit = self.system[-1]  # lowest is pure Bytes
        self.find_largest_unit()

    def _find_largest_unit_pos(self, num_bytes):
        """Find the proper unit and corresponding amount
        This implementation only works for positive number
        """
        factor = None
        for factor, unit in self.system:
            if num_bytes >= factor:
                break
        self.amount = num_bytes / factor

        # singular and plural for a tuple
        if isinstance(unit, tuple):
            singular, multiple = unit
            if self.amount == 1:
                unit = singular
            else:
                unit = multiple
        self.unit = unit

    def find_largest_unit(self):
        num_bytes = self.num_bytes
        pos_bytes = abs(num_bytes)
        sign = int(num_bytes >= 0) * 2 - 1  # sign function
        self._find_largest_unit_pos(pos_bytes)
        self.amount *= sign

    def __str__(self):
        return str(int(round(self.amount))) + self.unit

    def __format__(self, formatstr):
        if formatstr:
            return self.amount.__format__(formatstr) + self.unit
        else:
            return str(self)

    def __repr__(self):
        return str(self) + '<ByteSize amount={}>'.format(self.amount)

    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == other
        elif isinstance(other, ByteSize):
            return self.amount == other.amount
        else:
            return type(other)(self.amount) == other

    def __lt__(self, other):
        if isinstance(other, str):
            raise NotImplementedError(
                'Comparison between string and ByteSize not supported yet'
            )
        elif isinstance(other, ByteSize):
            return self.amount < other.amount
        else:
            return type(other)(self.amount) < other

    def __gt__(self, other):
        if isinstance(other, str):
            raise NotImplementedError(
                'Comparison between string and ByteSize not supported yet'
            )
        elif isinstance(other, ByteSize):
            return self.amount > other.amount
        else:
            return type(other)(self.amount) > other


def size(numbytes, system=None):
    """Human-readable file size.
    Using the traditional system, where a factor of 1024 is used::
    >>> size(10)
    '10B'
    >>> size(100)
    '100B'
    >>> size(2000000)
    '1M'
    Using the SI system, with a factor 1000::
    >>> size(10, system=si)
    '10B'
    >>> size(100, system=si)
    '100B'
    >>> size(1000, system=si)
    '1K'
    >>> size(2000000, system=si)
    '2M'
    """
    system = system or traditional
    byte_size = ByteSize(numbytes, system)
    return byte_size
