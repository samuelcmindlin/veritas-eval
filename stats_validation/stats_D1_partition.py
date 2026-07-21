import numpy as np
S = 0.10  # SESOI

def cells(L, U):
    hits = []
    if L >= S: hits.append(1)
    if (0 < L < S) and (U >= S): hits.append(2)
    if (L > 0) and (U < S): hits.append(3)
    if (L <= 0) and (U < S): hits.append(4)
    if (L <= 0) and (U >= S): hits.append(5)
    return hits

# Dense grid including exact boundary values 0 and SESOI, plus random floats
grid = np.unique(np.concatenate([
    np.linspace(-0.3, 0.4, 141),          # includes 0.0 and 0.10 exactly
    np.array([0.0, S, np.nextafter(0,1), np.nextafter(0,-1),
              np.nextafter(S,0), np.nextafter(S,1)]),
    np.random.default_rng(1).uniform(-0.3, 0.4, 4000)
]))

bad_none, bad_multi = [], []
present_mismatch = []
for L in grid:
    for U in grid:
        if U < L:  # invalid CI
            continue
        h = cells(L, U)
        if len(h) == 0: bad_none.append((L, U))
        if len(h) > 1: bad_multi.append((L, U, h))
        # D4: union of cells 1-3 must equal {L > 0}
        if (len(h) == 1) and ((h[0] in (1, 2, 3)) != (L > 0)):
            present_mismatch.append((L, U, h))

print("pairs with NO cell:", len(bad_none))
print("pairs with >1 cell:", len(bad_multi))
print("cells1-3 vs L>0 mismatches:", len(present_mismatch))

# Boundary case assignments
for L, U in [(0.0, 0.05), (0.0, 0.2), (S, 0.2), (S, S), (0.05, S), (0.0, S),
             (np.nextafter(0,1), S), (0.05, np.nextafter(S,0))]:
    print(f"L={L:.6g} U={U:.6g} -> cell {cells(L,U)}")

# Show that WITHOUT the L<=U precondition, cells 1 and 3 overlap
print("L=0.12,U=0.08 (invalid L>U):", cells(0.12, 0.08))
