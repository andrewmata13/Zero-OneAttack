# Session findings — index

Detailed notes behind the decisions in ../../CLAUDE.md. Copied from the working session's memory so the handoff is self-contained.

- [Zero-One ideas ledger](zero-one-ideas-ledger.md) — what has been tried and what the measurements said; check before proposing improvements.
- [PGD loss drives saturation](pgd-loss-drives-saturation.md) — the archive's "wrong" CrossEntropy loss is what makes the attack work; legacy_norand is the best known config.
- [Attack quality acceptance bar](attack-quality-acceptance-bar.md) — Cheetah must reach reward <= -1000; percentage-of-archived bands are too lenient.
- [Compare attacks at equal episode length](compare-attacks-at-equal-episode-length.md) — rewards are not comparable across maxlen; budget is per-segment.
- [Search regime depends on dimension](search-regime-depends-on-dimension.md) — "random ≈ SRACOS" holds only at dim 120, not with the block basis.
- [Attack budget and Adv_Traj](attack-budget-and-adv-traj.md) — the saved traces are the paper's high-budget (1500) results; budget is unrecorded and dominates.
- [Adv_Traj replay divergence](adv-traj-replay-divergence.md) — replaying saved trajectories reproduces Hopper/Walker exactly but not Cheetah/Ant.
- [MJX port Phase 0](mjx-port-phase0.md) — victim behavior transfers to modern MuJoCo (1.01); port viable, victims kept.
