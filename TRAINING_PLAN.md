# Training Plan: ML Bot + RL Bot

Goal: build a poker bot that can credibly win a **single multi-agent tournament match** against a field that includes CFR, MC200, GTO, and others — not just one that wins on average over many tournaments.

**Format context:** a match is a multi-way tournament (typically 5–7 players) with escalating blinds. Players bust out one at a time. Only the endgame (last 2 players) is heads-up. This means the bot needs to play well at full tables, short-handed, *and* heads-up — and it needs tournament-aware skills like survival, bubble play, and stack management.

This plan is the combined result of brainstorming with two chat agents (ChatGPT + Claude). They independently agreed on the biggest priorities, which is a good sign.

---

## Locked decisions (from brainstorm)

- **Real match format:** 6 players, winner-take-all, one shot, no info on the 5 opposing bots beforehand.
- **Training table size:** 6 seats (match the real format).
- **Payout implication:** 2nd place = 6th place = zero. So survival-for-survival's-sake is worth less than it looks. What matters is reaching heads-up *with chips* AND winning heads-up. Two separate skills, both trained.
- **ICM simplifies:** since winner-take-all, "tournament equity" = P(winning the whole thing). Much cleaner than multi-prize ICM. We estimate this via simulation during training instead of running full ICM math.
- **Multi-way vs heads-up framing:** not a contradiction. Train multi-way for survival + chip accumulation, *then* fine-tune heads-up specialists for the endgame (Step 13). Both phases matter.
- **Hardware:** Apple M5 Max — 18 CPU cores (~10x scaling measured), 40-core GPU via MPS (~8x speedup on models ≥2048 hidden), 64GB RAM. See `HARDWARE_BENCHMARK.md`.
- **Imitation target:** CFR bot (currently strongest). Regenerating the dataset on the M5.
- **Parallelization plan:**
  - Week 1 (data gen + AIVAT labeling + eval): use `multiprocessing.Pool(18)`. Embarrassingly parallel, free 10x.
  - Week 1 (CFR retraining): CFR needs special handling — not embarrassingly parallel because all workers update the same regret table. See "CFR parallelization" section below.
  - Week 4+ (PPO training loop): vectorized envs or async actors. Architectural choice, tackled when we get there — not front-loaded.

---

## CFR parallelization (optional future optimization — not required)

For the Week 1 CFR retrain we run **single-threaded overnight** — no parallelization needed. The time budget is "timeless" and single-threaded is the theoretical gold standard (no drift, clean convergence).

This section is reference material in case we want to speed up a *future* CFR retrain, not a prerequisite for Week 1.

CFR is different from data-gen or eval workloads. Workers can't just run independent tournaments and merge chip counts — they all need to update the *shared* regret table, and naive parallel writes will race and corrupt the data. Four real options if we ever revisit this:

**Option 1 — Per-worker regret tables, merged periodically.** Each of the 18 workers keeps its own regret dict, trains independently, and every N episodes the main process merges them (sum the regrets, re-normalize). Simplest to implement. Workers drift between merges, so there's a small quality loss per iteration (~5–10%), but merges every 100–500 episodes keep drift tight. Runs at ~8–10x wall-clock speedup.

**Option 2 — Shared regret table with a lock.** All workers write to the same `multiprocessing.Manager().dict()`. Correctness is perfect — no drift. But lock contention kills most of the parallel speedup. Not worth it in practice.

**Option 3 — Lock-free shared memory via numpy arrays.** Map regrets to fixed-size numpy arrays in shared memory, workers do atomic-ish updates. Fastest and most correct. Most complex to implement (requires re-keying the regret dict to integer indices).

**Option 4 — Ensemble of independent CFR runs.** Train 6–10 independent CFR models on separate workers (no merging, fully independent), then at inference time average their regret tables. This is how a lot of published poker research bots are actually trained. The independent convergence paths smooth out each other's blind spots. Often the highest final quality, but adds complexity at inference (bot has to load/merge multiple files).

**When to revisit:** only if we find the single-threaded CFR retrain is a real bottleneck *and* we want to iterate on CFR multiple times. For a one-shot retrain, single-threaded overnight is fine.
- **"AIVAT" naming:** technically a misnomer for what we're doing. The real name is "full-information equity shaping" — we peek at opponent cards during training and reward by actual equity. Works fine, just not literally the AIVAT paper's method.
- **Time budget:** timeless. Overnight / weekend runs are fine.

---

## The 3 things that matter most

Both agents agreed these are the real leverage points:

1. **Reward signal is the #1 problem.** Chip delta per hand is too noisy. Fold AA = 0 reward, shove 72o and suckout = big reward. The bot learns garbage.
2. **Warm-start with imitation, then RL, then league play.** Don't start RL from random weights.
3. **Features are too weak.** Especially missing: betting history encoding, and per-street opponent stats.

Everything below serves these three goals.

---

## Must-do steps (70% of the benefit)

### Step 1 — Generate a training dataset

Run thousands of tournaments with **only your strong bots** playing (cfr, mc200, gto, exploitative, icm). Log every decision to JSONL.

- Target: 500k–1M decision rows
- Players: `cfr,mc200,gto,exploitative,icm,smart` — 5–7 player tables (same size as real matches)
- Vary blind levels and stack depths: capture early-game (deep stacks), mid-game (medium), and late-game (short-stack, bubble, heads-up)
- Time: 1–2 hours of compute

This is the raw material for everything downstream. Without it, there's nothing to imitate. Crucially — generate data from **all stages of a tournament**, not just full-table play. Short-handed and heads-up decisions are very different and need their own training examples.

### Step 2 — Implement AIVAT-style reward (with ICM adjustment)

Replace raw chip delta with equity-based reward. This is the single biggest upgrade.

**The idea:** instead of rewarding the bot for chips it actually won, reward it based on the equity of its hand at the decision point.

- Call with 80% equity → "deserves" 80% of the pot, even if it loses the hand
- Call with 20% equity → "deserves" 20% of the pot, even if it sucks out

**How to implement:**

- At each decision, snapshot pot, stacks, hole cards, opponent hole cards, board
- At hand end, run a Monte Carlo rollout against **all remaining opponents' actual hands** (full-information during self-play training — you know everyone's cards, the bot doesn't)
- Reward = `EV_realized - EV_if_you_folded`

**Multi-way adjustment (important for your format):** in a tournament, equity-at-showdown isn't the only thing that matters. Chips aren't linear in tournament value — losing your last 100 chips is way worse than losing 100 when you have 1000. Wrap the AIVAT reward in an **ICM (Independent Chip Model) transform** so the reward reflects tournament equity, not raw chip equity.

- Early in the tournament (deep stacks): ICM ≈ linear, AIVAT is almost enough on its own
- Mid/late (short stacks, bubble, heads-up): ICM diverges sharply — a double-up is worth much less than losing it all
- You already have an ICM bot — reuse its math

Your existing equity calculator (used in MonteCarloBot) is most of the work already done. Wrap it into the training loop, then wrap that in the ICM transform.

**Validation checks:**

- Folding AA preflop should give strongly negative reward regardless of what cards come out
- Going all-in as short stack with marginal hands should be rewarded more in early stages than on the bubble (ICM pressure)

### Step 3 — Warm-start the RL bot from imitation

Don't train PPO from scratch. It wastes the first 50k episodes learning "folding every hand is bad."

- Train ML bot via supervised learning on the CFR bot's decisions from Step 1
- Filter to CFR-only decisions (`--filter_players P_cfr`) to clone the strongest bot specifically
- Copy the trained policy weights into the RL bot's policy network as initialization
- Now the RL bot plays decent poker before a single RL episode

**Target:** warm-started bot should roughly break even vs CFR heads-up. If it's getting crushed, something's wrong before starting RL.

### Step 4 — League training in full multi-way tournaments

Train against a rotating pool of opponents in the actual match format (5–7 players, escalating blinds, bust-out to heads-up), not just heads-up or fixed 4-player.

**Pool composition (approx):**

- 25% recent snapshots of the RL bot itself
- 25% CFR / GTO variants
- 20% MC200 and other heuristic bots
- 15% exploitative / opponent-model / ICM bots
- 15% "stress" styles (nit, maniac, random)

**Match structure during training:**

- 5–7 player tables (match your real eval format)
- Escalating blinds (same schedule as real matches)
- Bots bust out naturally — the RL bot has to learn full-table → short-handed → heads-up transitions
- Randomize seat position each episode so the bot sees all positions
- Randomize starting stacks across episodes (vary tournament "stage" the bot drops into)

**Rules:**

- Snapshot the current RL bot every 500–1000 episodes, add to pool
- Each match, fill the other seats by random sampling from the pool (with replacement, weighted toward opponents causing recent losses)
- Use `train_multi_deep_rl_bot.py` as the starting point (already multi-opponent — just expand it)

This avoids two classic failure modes:

- Cycling (bot chases its own tail in rock-paper-scissors strategies)
- Degenerate equilibria (bot and its mirror converge to weird exploitable patterns)

And it ensures the bot learns the three distinct skills tournament play requires: **deep-stack full-ring play, short-handed / bubble play, and heads-up endgame.**

---

## Should-do steps (20% of the benefit)

### Step 5 — Add betting history encoding

Whether a pot got to $200 via `check-check-bet-raise-call` vs `bet-raise-call-call` matters hugely for hand reading. Right now this is thrown away.

Add a small GRU or transformer encoder over the sequence of (player, action, size) tuples in the current hand.

### Step 6 — Expand opponent stats to per-street, per-opponent

Right now: 3 scalars averaged across all opponents from the last 10 actions. Way too coarse for multi-way play where different opponents need different reads.

Change to **per-opponent, per-street stats**:

- For each active opponent: preflop aggression, flop aggression, turn aggression, river aggression (separately)
- Bet-sizing tendencies per street
- Showdown frequency
- VPIP and PFR separately

This matters much more in multi-way than heads-up — you might want to call a tight player's raise but fold the exact same hand to a maniac's raise at the same table. The current "average across opponents" feature can't capture that.

Expand from 3 total features to ~10 per opponent × up to 6 opponents = 60 features (or use attention/pooling to keep it manageable).

### Step 7 — Drop the fixed 10% exploration

Both agents flagged this. Forced random moves on top of a stochastic policy is double-dipping and hurts credit assignment.

- Remove the hard epsilon-greedy override
- Use PPO's built-in entropy bonus instead (start coef ~0.01, anneal to ~0.001)

### Step 8 — Fix the feature mismatch bug

The README already flags this: `ml_bot.py` maps `"BB": 0.3` at inference time, but `train_ml_bot.py` maps `"BB": 0.5`. Training and inference should match exactly.

Quick one-line fix, but easy to miss.

### Step 9 — Evaluate properly (tournament-aware)

Stop using 50-tournament tests. Poker variance is brutal, and **tournament variance is even worse than cash-game variance** because outcomes are binary (you win or bust).

- **Run 500–1000 tournaments minimum** for any checkpoint comparison
- Track: tournament win rate (1st place %), ITM rate (in-the-money %), average finishing position, hands survived
- Report Wilson confidence intervals on win rate — random in a 7-player field ≈ 14%, so anything <20% is noise
- Measure exploitability periodically (train a dedicated exploiter against the current bot, see how much it wins)
- For AIVAT-style per-hand evals, use bb/100 on individual hands (finer signal, less variance than tournament outcomes)
- Randomize seating and starting-stage conditions to avoid positional bias

---

## Nice-to-have steps (10% of the benefit)

### Step 10 — Recurrent opponent model

Add a GRU that persists across hands within a single match. Builds an implicit opponent model in real time. Matters most for single-match strength.

### Step 11 — Continuous bet sizing

6 action classes (fold/check/call/small/med/large) is too coarse. Pros agonize over 1/3 pot vs 2/3 pot.

Either:

- Expand to 9–12 discrete buckets, or
- Hybrid head: discrete {fold, check, call, raise, all-in} + continuous sizing output when raise is chosen

### Step 12 — Randomize stack depths in training

Strategy at 200bb deep is very different from 20bb. Training at only one stack depth gives you a fragile bot. Randomize starting stacks 50–200bb.

### Step 13 — Targeted fine-tuning

Once you have a solid league-trained bot, clone it three times. Fine-tune each copy against one specific target (CFR / MC200 / GTO) with KL regularization back to the league policy (β ~ 0.01) to prevent overfitting.

At match time, pick the right specialist based on opponent.

### Step 14 — Test-time search (DeepStack-style)

At decision time, use the policy as a prior and do shallow lookahead with the value network for leaf evaluation. Huge boost for single-match strength but meaningful implementation work.

### Step 15 — Deep CFR / ReBeL

If you want to go nuclear: swap PPO for Deep CFR or ReBeL. These are purpose-built for imperfect-information games and have much better theoretical grounding for poker than PPO. Big codebase change. Only consider after everything else is working.

---

## Concrete timeline

### Week 1 — Infrastructure

- **Overnight background run:** retrain CFR for 6-seat tables (current profile is 4-max). Save to a new profile file, don't overwrite existing. Runs unattended while we work on everything else below.
- Parallelize data-gen + eval with `Pool(18)` before anything else
- Run tournaments to generate dataset at 6-seat tables (Step 1), using the newly retrained CFR as the main imitation target
- Implement full-info equity reward with ICM = P(winning tournament) (Step 2)
- Validate reward: fold AA test case, short-stack shove test case
- Build proper eval harness: 500–1000 tournament matches with Wilson CIs on win rate
- Randomize stack depths in training envs
- Fix BB position-encoding bug (Step 8)
- Drop fixed 10% exploration (Step 7) — it's a one-liner, do it now

### Week 2 — Architecture upgrade

- Expand features: card embeddings, action-history encoder, per-street opponent stats (Steps 5, 6)
- Optional: add continuous bet-sizing head (Step 11)
- Optional: wire in opponent-modeling GRU even if unused initially (Step 10)

### Week 3 — Supervised warm-start

- Train ML bot on CFR's decisions with AIVAT labels (Step 3)
- Target: warm-started bot breaks even vs CFR heads-up
- If getting crushed, stop and debug before starting RL

### Week 4–5 — PPO vs fixed pool (multi-way first)

- Parallelize the PPO training loop — vectorized envs or async actors
- Train 6-seat multi-way against `{MC200, CFR, GTO, heuristic, warm-started-frozen}` (~20–30M hands)
- Goal here: survival + chip accumulation at the real table size. Heads-up specialists come in Week 8.
- Monitor: reward/hand going up, entropy not collapsing, KL to warm-start not exploding
- Checkpoint every 500k hands

### Week 6–7 — League self-play

- Add checkpoint pool, prioritized sampling toward recent (Step 4)
- Use NFSP-style mixing (play against average of past policies, not just latest)
- Track exploitability; stop when it plateaus

### Week 8 — Heads-up specialists (the endgame)

- Clone the league-trained bot, fine-tune specialists for heads-up vs each archetype (tight, aggro, GTO-ish, etc.)
- KL regularization back to league policy (β ~ 0.01) so specialists don't drift too far
- On gameday, since we don't know who's left heads-up, we pick a specialist based on what we *observed* from the opponent during the multi-way phase

### Week 9 — Final eval

- 10k-hand matches vs each target
- Report win rates with confidence intervals
- If not winning, most likely culprit is still reward signal or warm-start quality — go back and verify AIVAT first

---

## Common mistakes to avoid

In rough order of how often they bite people:

- Training on chip delta and wondering why nothing converges
- Self-play from scratch (gets a bot great at beating its past self, terrible at everything else)
- Not tracking exploitability (you won't know if you're overfitting to the league)
- Evaluating on too few hands (need ~10k+ to distinguish signal from noise)
- Ignoring stack depth / SPR (train at one depth → fragile bot)
- Heads-up-only training when final eval is multi-way (strategies transfer poorly — this applies directly to your format)
- Training on full-table play only and hoping heads-up skills emerge (they won't; heads-up needs its own training data)
- Ignoring ICM / tournament equity and treating chips as linear value
- Not normalizing reward by big blind across escalating blind levels
- PPO clipping ratio too tight (default 0.2 is often too restrictive for poker; try 0.3)
- Using "winning hands only" in supervised training — survivorship bias, result bias
- Weak value targets — if the value model is poor, shaped rewards become garbage

---

## The honest bottom line

If you do Steps 1–4 (the Must-do section), you'll go from "random noise" to "competitive with CFR." That alone is probably 70% of what you want.

Steps 5–9 bring polish and robustness. Steps 10–15 push toward "superhuman" but have diminishing returns and much bigger engineering cost.

**Single most impactful change:** implement AIVAT reward (Step 2).

**Single most impactful pipeline change:** imitation warm-start → value-shaped PPO → league training (Steps 3–4).

**Single most impactful architecture change:** replace the flat 26-feature summary with richer card + board + betting-history + opponent-summary encoding (Steps 5–6).
