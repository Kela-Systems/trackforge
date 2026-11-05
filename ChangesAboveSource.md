# Changes above upstream

Kela's branch `kela/v0.5.1` on top of upstream release 0.5.1 (`16fb8bf`, "Merge pull request #218 … release/0.5.1"; `src/` identical to the crates.io 0.5.1 package). `main` mirrors upstream untouched.

To move to a new upstream release: rebase this branch onto the release commit, replay the changes below, run `cargo test` (includes `tests/cascade_parity.rs`), and update the pinned `rev` in Kela's `hub/Cargo.toml`.

| Commit | Change | Files | Why |
|---|---|---|---|
| `ff40f47` | `byte_cascade::run` returns the unmatched high detections; new `byte_cascade::spawn` starts tracks from them. No behaviour change. | `common/byte_cascade.rs`, `byte_track/mod.rs`, `botsort/mod.rs` | Lets a tracker match unconfirmed tracks against the leftovers before new tracks start. |
| `af1893b` | New tracks (after frame 1) stay unconfirmed until their next match, in both ByteTrack and BoT-SORT: out of both cascade stages, not predicted, match only leftover high detections at `1 − IoU·score ≤ 0.7` (+ half gated cosine with ReID in BoT-SORT), dropped on a miss. New `byte_cascade::confirm`. Tests adjusted where they expected an id on a track's first frame; new BoT-SORT tests. | `common/byte_cascade.rs`, `byte_track/mod.rs`, `botsort/mod.rs`, `botsort/README.md` | Matches the reference implementations (ifzhang/ByteTrack, BoT-SORT/boxmot). Without it every one-frame false positive gets an id and duplicates steal established tracks' detections. Candidate upstream PR. |
