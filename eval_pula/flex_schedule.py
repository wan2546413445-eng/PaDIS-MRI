"""External P/D update schedule, independent of torch and model code."""
import re
from pathlib import Path


def make_schedule(n, pattern=None, blocks=None, schedule_file=None):
    """1-based inclusive ranges: 1-20:PPPPP;21-47:PPPPD.

    If pattern is absent, ranges must cover all N levels. No implicit steps.
    Each level may have a different number of slots; all must be explicit.
    """
    if n < 1:
        raise ValueError('N must be positive')
    if blocks is not None and schedule_file is not None:
        raise ValueError('Use --blocks or --schedule_file, not both')
    def check(p):
        p = p.strip().upper()
        if not re.fullmatch('[PD]+', p):
            raise ValueError(f'Invalid pattern: {p!r}; only P and D allowed')
        return p
    rows = [check(pattern) if pattern is not None else None] * n
    if schedule_file is not None:
        blocks = ';'.join(line.split('#', 1)[0].strip() for line in
                          Path(schedule_file).read_text().splitlines()
                          if line.split('#', 1)[0].strip())
    used = set()
    if blocks:
        for block in blocks.split(';'):
            m = re.fullmatch(r'\s*(\d+)(?:\s*-\s*(\d+))?\s*:\s*([PDpd]+)\s*', block)
            if m is None:
                raise ValueError(f'Invalid block: {block!r}')
            lo, hi = int(m[1]), int(m[2] or m[1])
            if not 1 <= lo <= hi <= n:
                raise ValueError(f'Range {lo}-{hi} outside 1..{n}')
            for i in range(lo - 1, hi):
                if i in used:
                    raise ValueError(f'Overlapping blocks at level {i+1}')
                rows[i] = check(m[3]); used.add(i)
    if any(row is None for row in rows):
        raise ValueError('Unassigned levels; use --pattern or cover every level in --blocks')
    return rows


def make_sigmas(n, sigma_max, sigma_min, kind, rho=7.0):
    if not 0 < sigma_min < sigma_max or n < 1 or rho <= 0:
        raise ValueError('Invalid noise schedule')
    if kind == 'karras':
        if n < 2:
            raise ValueError('Karras schedule requires N>=2')
        grid = [(sigma_max ** (1/rho) + i/(n-1)*(sigma_min ** (1/rho)-sigma_max ** (1/rho))) ** rho for i in range(n)]
    elif kind == 'exponential':
        grid = [sigma_max * (sigma_min / sigma_max) ** (i/n) for i in range(1,n+1)]
    else:
        raise ValueError(kind)
    grid[-1] = sigma_min
    return grid


def counts(rows):
    p = sum(r.count('P') for r in rows)
    d = sum(r.count('D') for r in rows)
    return dict(pula_updates=p, dps_updates=d, logical_updates=p+d,
                logical_nfe=p+d, network_backprops=d, cg_solves=p+1)
