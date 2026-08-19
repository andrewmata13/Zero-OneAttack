---
name: attack-quality-acceptance-bar
description: "Any cheaper/faster Zero-One variant must land within 10-20% of the archived attack's damage to be acceptable"
metadata: 
  node_type: memory
  type: project
  originSessionId: d92ad1e9-cda4-4038-b191-a04985b9a7d8
  modified: 2026-08-17T02:38:11.827Z
---

Speed and budget reductions to the Zero-One attack are only acceptable if attack strength
holds up. **The agreed bar for Half Cheetah is reward <= -1000 over 1000 steps.**

Do not use a percentage-of-archived band as the bar. It is misleadingly lenient near the
top: most of the damage (7164 -> ~0) is easy, and everything that distinguishes attacks
lives in the last few percent where reward goes negative. Measured on Cheetah PPO,
Zhang/SA-RL already sits at 92.0% of archived damage, so a "within 20%" (>=80%) or even
"within 10%" (>=90%) bar would pass attacks that lose to the baseline being beaten.

Reference for Half Cheetah PPO over 1000 steps: clean 7163.9, archived Zero-One -1336.5
(budget 1500), Zhang/SA-RL -652.9 (92.0%). The -1000 target corresponds to 96.0% of
archived damage, which clears Zhang.

Damage percentage, when reported, is measured from clean rather than as a raw reward ratio,
since rewards change sign under attack:

    pct = (clean - achieved) / (clean - archived) * 100

`opt_attack/run_attack.py` carries the target in its TARGET table and prints a PASS/FAIL verdict
automatically; override per run with --target.

**Why:** Andrew set this bar explicitly — the attack's value is that it is the strongest
known attack on RL dynamical systems, so a cheap variant that gives that up is not useful
even for the retraining loop.

**How to apply:** Report the damage percentage for every configuration change, and compare
only at equal episode length — see [[compare-attacks-at-equal-episode-length]].
