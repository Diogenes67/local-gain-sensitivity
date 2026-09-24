"""Random-transition m-th-order Markov sources over a small vocabulary, entropy matched.

A source is a transition table P[s, x] over the V^m contexts s = (x_{t-m}, ..., x_{t-1}) (state index
s = sum_j x_{t-m+j} V^(m-1-j), so the most recent token is the least significant digit). Rows are drawn
from Dirichlet(alpha) and tempered, P ∝ P0^(1/tau), with tau chosen by bisection so that the stationary
conditional entropy H(X_t | X_{t-m..t-1}) equals H_TARGET nats for every order and seed. Rows have full
support, so every chain is irreducible and aperiodic. Sequences start from the stationary distribution
over contexts, so no burn-in is needed and every position of a sequence has a full-order context in the
source (the model only sees the sequence, so its own floor at positions t < m is higher; the learned-gap
statistic uses positions t >= 8, valid for every order <= 8).
"""
import numpy as np

V_DEFAULT, H_TARGET, ALPHA = 8, 1.0, 0.5   # ALPHA is the Dirichlet parameter of the dense structure only


def stationary(P, V, m, tol=1e-12, max_iter=5000):
    S = P.shape[0]; base = V ** (m - 1)
    s = np.arange(S); nxt = np.stack([(s % base) * V + x for x in range(V)], axis=1)   # S x V next-state index
    pi = np.full(S, 1.0 / S)
    for it in range(max_iter):
        new = np.zeros(S)
        for x in range(V):
            new += np.bincount(nxt[:, x], weights=pi * P[:, x], minlength=S)
        new /= new.sum()
        if np.abs(new - pi).sum() < tol:
            pi = new; break
        pi = new
    return pi, it + 1


def cond_entropy(P, pi):
    return float(-(pi[:, None] * P * np.log(P)).sum())


def temper(P0, tau):
    L = np.log(P0) / tau; L -= L.max(axis=1, keepdims=True)
    P = np.exp(L); P /= P.sum(axis=1, keepdims=True)
    return P


def floors_by_order(P, pi, V, m):
    """H(X_t | last r tokens) for r = 0..m under the stationary law: the best loss a predictor that reads only
    the last r tokens can reach. r = 0 is the marginal entropy, r = m the conditional entropy of the source."""
    S = P.shape[0]; s = np.arange(S); out = []
    for r in range(m + 1):
        g = s % (V ** r)                                   # suffix-r index of every state
        joint = np.stack([np.bincount(g, weights=pi * P[:, x], minlength=V ** r) for x in range(V)], axis=1)   # P(suffix_r, x)
        marg = joint.sum(axis=1, keepdims=True); cond = joint / np.clip(marg, 1e-300, None)
        out.append(float(-(joint * np.log(np.clip(cond, 1e-300, None))).sum()))
    return out


def raw_table(V, m, seed, alpha, structure):
    """Untempered transition rows. dense: independent Dirichlet(alpha) rows, one per context (no structure
    below order m, so the gradient signal for a partial context of r < m tokens is of order V^-(m-r)/2 and
    the source is not learned at m >= 5 in the budget). hier: logits are a sum over r = 1..m of independent
    standard-normal tables indexed by the last r tokens, so every order contributes and the full order-m
    table is needed to reach the floor; the analogue of a back-off source with random statistics."""
    rng = np.random.default_rng(1_000_003 * m + 7919 * seed + 11)
    S = V ** m
    if structure == "dense":
        P0 = rng.dirichlet(np.full(V, alpha), size=S)
    else:
        s = np.arange(S); logits = np.zeros((S, V))
        for r in range(1, m + 1):
            G = rng.standard_normal((V ** r, V)); logits += G[s % (V ** r)]
        logits -= logits.max(axis=1, keepdims=True); P0 = np.exp(logits); P0 /= P0.sum(axis=1, keepdims=True)
    P0 = np.clip(P0, 1e-12, None); P0 /= P0.sum(axis=1, keepdims=True)
    return P0


def _match(P0, V, m, H_target):
    lo, hi = -4.0, 4.0     # log tau; H is increasing in tau (tau -> 0 deterministic rows, tau -> inf uniform rows)
    for _ in range(40):
        mid = 0.5 * (lo + hi); P = temper(P0, np.exp(mid)); pi, _ = stationary(P, V, m); H = cond_entropy(P, pi)
        if H < H_target: lo = mid
        else: hi = mid
    tau = np.exp(0.5 * (lo + hi)); P = temper(P0, tau); pi, n_it = stationary(P, V, m); H = cond_entropy(P, pi)
    marg = np.zeros(V)
    for x in range(V): marg[x] = (pi * P[:, x]).sum()
    return P, pi, tau, H, marg, float(-(marg * np.log(marg)).sum()), n_it


MARGINAL_SLACK = 0.10   # accept a source only if its marginal entropy is within this of ln V, so mutual information is matched too


def make_source(V=V_DEFAULT, m=1, seed=0, alpha=ALPHA, H_target=H_TARGET, structure="hier"):
    """Draw the table, temper it to H_target, and (low orders have few states, so the stationary marginal can be
    skewed) redraw with a sub-seed until the marginal entropy is within MARGINAL_SLACK of ln V."""
    S = V ** m
    for attempt in range(500):
        P0 = raw_table(V, m, seed + 100_000 * attempt, alpha, structure)
        pi0, _ = stationary(P0, V, m); H0 = cond_entropy(P0, pi0)
        P, pi, tau, H, marg, H_marg, n_it = _match(P0, V, m, H_target)
        if H_marg >= np.log(V) - MARGINAL_SLACK: break
    return dict(V=V, m=m, seed=seed, alpha=alpha, H_target=H_target, structure=structure, P=P.astype(np.float64), pi=pi, tau=float(tau), H_cond=H,
                H_cond_untempered=H0, H_marginal=H_marg, mutual_information=H_marg - H, marginal=marg, n_states=S, stationary_iters=int(n_it),
                draw_attempts=attempt + 1, floors_by_order=floors_by_order(P, pi, V, m))


def generate(src, n_seqs, seq_len, seed):
    """n_seqs sequences of seq_len tokens; initial context drawn from the stationary distribution."""
    P, pi, V, m = src["P"], src["pi"], src["V"], src["m"]; base = V ** (m - 1)
    rng = np.random.default_rng(seed)
    s = rng.choice(len(pi), size=n_seqs, p=pi)
    # the first m tokens are the digits of the initial context (most recent last)
    ctx = np.zeros((n_seqs, m), dtype=np.int64); ss = s.copy()
    for j in range(m - 1, -1, -1):
        ctx[:, j] = ss % V; ss //= V
    out = np.zeros((n_seqs, seq_len), dtype=np.int64); out[:, :m] = ctx[:, :seq_len]
    cum = np.cumsum(P, axis=1); cum[:, -1] = 1.0 + 1e-9
    for t in range(m, seq_len):
        u = rng.random(n_seqs)
        x = (u[:, None] > cum[s]).sum(axis=1)
        out[:, t] = x; s = (s % base) * V + x
    return out


def state_index(seqs, V, m, t):
    """context index for the token at position t (needs t >= m): digits seqs[:, t-m:t], most recent least significant."""
    s = np.zeros(seqs.shape[0], dtype=np.int64)
    for j in range(t - m, t): s = s * V + seqs[:, j]
    return s


def floor_per_position(src, seqs):
    """-ln P(x_t | x_{t-m..t-1}) under the true source for every position t >= m; nan before."""
    P, V, m = src["P"], src["V"], src["m"]; n, T = seqs.shape
    f = np.full((n, T), np.nan)
    for t in range(m, T):
        s = state_index(seqs, V, m, t); f[:, t] = -np.log(P[s, seqs[:, t]])
    return f


if __name__ == "__main__":
    import time, sys
    structure = sys.argv[1] if len(sys.argv) > 1 else "hier"
    for m in (1, 2, 3, 4, 5, 6):
        t0 = time.time(); src = make_source(m=m, seed=0, structure=structure)
        seqs = generate(src, 800, 128, seed=1); fl = floor_per_position(src, seqs)
        print(f"{structure} m={m}: states {src['n_states']:7d} tau {src['tau']:.3f} H0 {src['H_cond_untempered']:.3f} -> H {src['H_cond']:.5f}  "
              f"H_marg {src['H_marginal']:.3f} MI {src['mutual_information']:.3f} draws {src['draw_attempts']}  floors by order {np.round(src['floors_by_order'], 3).tolist()}  "
              f"held floor(t>=8) {np.nanmean(fl[:, 8:]):.4f}  ({time.time() - t0:.1f} s)")
