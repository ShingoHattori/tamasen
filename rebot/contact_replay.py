"""Replay a move trace (run/arm_trace_last.jsonl) through arm_state.ContactMonitor (read only, no bus).

    python contact_replay.py ../run/arm_trace_last.jsonl            # would the contact stop have fired?
    python contact_replay.py TRACE --wall 2:200 --wall 3:15:soft    # plus synthetic contacts on joint2 / joint3

A synthetic contact starts when the joint has covered 40 % of its planned travel. "wall" (default): the joint
stops there and the motor torque grows by K N m/rad times the plan's further advance (K ~ the position loop's
stiffness, ~200 N m/rad on joint2/3). "soft": the joint keeps moving at half the planned speed into a spring
of K N m/rad. Thresholds: rebot/mit_tuning.yaml "contact:" (as the arm service reads them at engage).
"""
import argparse
import json

import numpy as np

import arm_state as A


def load(path):
    L = [json.loads(x) for x in open(path)]
    head, rows = L[0], [r for r in L[1:] if r["state"] == "moving"]
    J = head["joints"]
    t = np.array([r["t"] for r in rows])
    P = np.array([r["plan"] for r in rows])
    Q = np.array([r["q"] for r in rows])
    tq = np.array([[x or 0.0 for x in r["torq"]] for r in rows])
    ff = np.array([[r["ff"].get(j, {}).get("tau_ff", 0.0) + r["ff"].get(j, {}).get("fric", 0.0) for j in J] for r in rows])
    return head, t, P, Q, tq, ff


def run(params, t, P, Q, tq, ff, lo, hi):
    cm = A.ContactMonitor(params)
    V = np.gradient(P, t, axis=0) if len(t) > 1 else np.zeros_like(P)
    for k in range(len(t)):
        active = (P[k] - lo > A.RES_LIMIT_MARGIN) & (hi - P[k] > A.RES_LIMIT_MARGIN)
        hit = cm.step(float(t[k]), tq[k], ff[k], Q[k], V[k], active)
        if hit:
            return k, hit, cm.peak
    return None, None, cm.peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="+")
    ap.add_argument("--contact", default="{}", help='override, e.g. \'{"spring_nm_per_rad": [10, 15, 15, 5, 5, 5]}\'')
    ap.add_argument("--wall", action="append", default=[], help="joint:K[:soft], e.g. 2:200 or 3:15:soft")
    a = ap.parse_args()
    tuning, err = A.ArmService.load_tuning()
    if err:
        raise SystemExit(err)
    params = tuning["contact"] | json.loads(a.contact)
    g = A.Gravity()
    lo = np.array([-np.pi] * 6); hi = np.array([np.pi] * 6)
    if g.error is None:
        for i, n in enumerate(g.names):
            if n in A.ARM:
                jid = g.model.joints[i + 1].idx_q
                lo[A.ARM.index(n)] = g.model.lowerPositionLimit[jid]; hi[A.ARM.index(n)] = g.model.upperPositionLimit[jid]
    print("contact:", json.dumps(params))
    for path in a.trace:
        head, t, P, Q, tq, ff = load(path)
        print(f"\n{path}: {head.get('path')} {head.get('time')}  moving {t[-1] - t[0]:.1f} s")
        k, hit, peak = run(params, t, P, Q, tq, ff, lo, hi)
        print("  as recorded:", f"STOP at {t[k]:.2f} s: {hit}" if hit else "no stop",
              "| peak rise / limit", [round(x, 2) for x in peak])
        for w in a.wall:
            parts = w.split(":")
            i, K, mode = int(parts[0]) - 1, float(parts[1]), (parts[2] if len(parts) > 2 else "wall")
            span = P[-1, i] - P[0, i]
            if abs(span) < 0.03:
                print(f"  joint{i + 1} {mode} K={K:g}: joint travels only {abs(span) * 1000:.0f} mrad in this move")
                continue
            s = np.sign(span)
            k0 = int(np.searchsorted(np.abs(P[:, i] - P[0, i]), 0.4 * abs(span)))
            adv = np.maximum((P[k0:, i] - P[k0, i]) * s, 0.0)
            Q2, tq2 = Q.copy(), tq.copy()
            if mode == "soft":
                Q2[k0:, i] = Q[k0, i] + s * 0.5 * adv
                tq2[k0:, i] += s * K * 0.5 * adv
            else:
                Q2[k0:, i] = Q[k0, i]
                tq2[k0:, i] += s * K * adv
            k, hit, _ = run(params, t, P, Q2, tq2, ff, lo, hi)
            if hit is None:
                print(f"  joint{i + 1} {mode} K={K:g}: MISSED")
            else:
                print(f"  joint{i + 1} {mode} K={K:g}: stop {t[k] - t[k0]:.2f} s after contact, plan "
                      f"{abs(P[k, i] - P[k0, i]) * 1000:.0f} mrad past it | {hit}")


if __name__ == "__main__":
    main()
