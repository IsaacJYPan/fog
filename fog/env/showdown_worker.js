"use strict";

/**
 * Persistent, serverless Pokemon Showdown worker.
 *
 * Protocol: one JSON object per stdin line, one JSON response per stdout line. The worker imports
 * the pinned local Showdown simulator directly; it does not start an HTTP server or use sockets.
 * Exact chance resolution uses controlled replay: every finite random request becomes an explicit
 * branch, and the same serialized root is replayed with each selected random prefix.
 */

const crypto = require("node:crypto");
const path = require("node:path");
const readline = require("node:readline");

function parseArgs(argv) {
  const args = {showdownRoot: null};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--showdown-root") {
      args.showdownRoot = argv[++i];
    } else {
      throw new Error(`unknown argument: ${argv[i]}`);
    }
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));
const showdownRoot = path.resolve(
  args.showdownRoot || path.join(__dirname, "..", "..", "server", "pokemon-showdown")
);
const {Battle, PRNG} = require(path.join(showdownRoot, "dist", "sim"));
const {extractChannelMessages} = require(path.join(showdownRoot, "dist", "sim", "battle"));
const {BattleActions} = require(path.join(showdownRoot, "dist", "sim", "battle-actions"));
const sessions = new Map();

function absBigInt(value) {
  return value < 0n ? -value : value;
}

function gcd(a, b) {
  a = absBigInt(a);
  b = absBigInt(b);
  while (b !== 0n) {
    [a, b] = [b, a % b];
  }
  return a;
}

function fraction(numerator, denominator = 1n) {
  numerator = BigInt(numerator);
  denominator = BigInt(denominator);
  if (denominator === 0n) throw new RangeError("probability denominator cannot be zero");
  if (denominator < 0n) {
    numerator = -numerator;
    denominator = -denominator;
  }
  const divisor = gcd(numerator, denominator);
  return {n: numerator / divisor, d: denominator / divisor};
}

function addFractions(left, right) {
  return fraction(left.n * right.d + right.n * left.d, left.d * right.d);
}

function multiplyFractions(left, right) {
  return fraction(left.n * right.n, left.d * right.d);
}

function divideFractions(left, right) {
  if (right.n === 0n) throw new RangeError("cannot divide by zero probability");
  return fraction(left.n * right.d, left.d * right.n);
}

function compareFractions(left, right) {
  const difference = left.n * right.d - right.n * left.d;
  return difference < 0n ? -1 : difference > 0n ? 1 : 0;
}

function fractionNumber(value) {
  return Number(value.n) / Number(value.d);
}

function fractionJSON(value) {
  return {
    numerator: value.n.toString(),
    denominator: value.d.toString(),
    decimal: fractionNumber(value),
  };
}

class ChanceBranch extends Error {
  constructor(index, key, outcomes) {
    super(`unresolved chance request ${index}: ${key}`);
    this.name = "ChanceBranch";
    this.index = index;
    this.key = key;
    this.outcomes = outcomes;
  }
}

/** Drop-in battle PRNG whose decisions come from a finite replay prefix. */
class EnumeratingPRNG {
  static ZERO_SEED = `sodium,${"0".repeat(64)}`;

  constructor(prefix = []) {
    this.prefix = prefix;
    this.cursor = 0;
    this.startingSeed = EnumeratingPRNG.ZERO_SEED;
  }

  getSeed() {
    return EnumeratingPRNG.ZERO_SEED;
  }

  clone() {
    const cloned = new EnumeratingPRNG(this.prefix);
    cloned.cursor = this.cursor;
    return cloned;
  }

  decide(key, outcomes) {
    if (outcomes.length === 0) throw new RangeError(`chance request has no outcomes: ${key}`);
    if (outcomes.length === 1) return outcomes[0].value;

    const index = this.cursor++;
    if (index >= this.prefix.length) {
      throw new ChanceBranch(index, key, outcomes);
    }
    const selected = this.prefix[index];
    if (selected.key !== key) {
      throw new Error(
        `chance replay diverged at ${index}: expected ${selected.key}, observed ${key}`
      );
    }
    const outcome = outcomes.find(candidate => candidate.choice === selected.choice);
    if (!outcome) {
      throw new Error(`invalid replay choice ${selected.choice} for ${key}`);
    }
    return outcome.value;
  }

  random(from, to) {
    if (from === undefined) {
      throw new Error(
        "unbounded random() is not exactly enumerable; no sampled fallback was used"
      );
    }

    let low;
    let high;
    if (to === undefined || !to) {
      low = 0;
      high = Math.floor(from);
    } else {
      low = Math.floor(from);
      high = Math.floor(to);
    }
    if (!Number.isSafeInteger(low) || !Number.isSafeInteger(high) || high <= low) {
      throw new RangeError(`invalid bounded random request [${low}, ${high})`);
    }

    const count = high - low;
    const weight = fraction(1n, BigInt(count));
    const outcomes = [];
    for (let value = low; value < high; value++) {
      outcomes.push({choice: value, value, weight});
    }
    return this.decide(`integer:${low}:${high}`, outcomes);
  }

  randomChance(numerator, denominator) {
    numerator = Math.floor(numerator);
    denominator = Math.floor(denominator);
    if (!Number.isSafeInteger(numerator) || !Number.isSafeInteger(denominator)) {
      throw new RangeError("randomChance requires integer arguments");
    }
    if (denominator <= 0 || numerator < 0 || numerator > denominator) {
      throw new RangeError(`invalid randomChance(${numerator}, ${denominator})`);
    }
    if (numerator === 0) return false;
    if (numerator === denominator) return true;
    return this.decide(`chance:${numerator}:${denominator}`, [
      {
        choice: 0,
        value: false,
        weight: fraction(BigInt(denominator - numerator), BigInt(denominator)),
      },
      {choice: 1, value: true, weight: fraction(BigInt(numerator), BigInt(denominator))},
    ]);
  }

  sample(items) {
    if (!Array.isArray(items) || items.length === 0) {
      throw new RangeError("cannot sample an empty or non-array collection");
    }
    return items[this.random(items.length)];
  }

  shuffle(items, start = 0, end = items.length) {
    while (start < end - 1) {
      const nextIndex = this.random(start, end);
      if (start !== nextIndex) {
        [items[start], items[nextIndex]] = [items[nextIndex], items[start]];
      }
      start++;
    }
  }
}

/** Ordinary Showdown PRNG plus a semantic transcript of every battle-time random answer. */
class RecordingPRNG {
  constructor(delegate, transcript = []) {
    this.delegate = delegate;
    this.transcript = transcript;
    this.startingSeed = delegate.startingSeed;
  }

  getSeed() {
    return this.delegate.getSeed();
  }

  clone() {
    return new RecordingPRNG(this.delegate.clone(), this.transcript);
  }

  record(key, choice, value) {
    this.transcript.push({key, choice});
    return value;
  }

  random(from, to) {
    const value = this.delegate.random(from, to);
    if (from === undefined) {
      throw new Error("session used unsupported unbounded random() during battle construction/play");
    }
    let low;
    let high;
    if (to === undefined || !to) {
      low = 0;
      high = Math.floor(from);
    } else {
      low = Math.floor(from);
      high = Math.floor(to);
    }
    if (high - low === 1) return value;
    return this.record(`integer:${low}:${high}`, value, value);
  }

  randomChance(numerator, denominator) {
    const value = this.delegate.randomChance(numerator, denominator);
    if (numerator === 0 || numerator === denominator) return value;
    return this.record(
      `chance:${Math.floor(numerator)}:${Math.floor(denominator)}`,
      value ? 1 : 0,
      value
    );
  }

  sample(items) {
    if (!Array.isArray(items) || items.length === 0) {
      throw new RangeError("cannot sample an empty or non-array collection");
    }
    return items[this.random(items.length)];
  }

  shuffle(items, start = 0, end = items.length) {
    while (start < end - 1) {
      const nextIndex = this.random(start, end);
      if (start !== nextIndex) {
        [items[start], items[nextIndex]] = [items[nextIndex], items[start]];
      }
      start++;
    }
  }
}

/** Replay a complete previously recorded prefix while constructing a candidate private world. */
class ForcedTranscriptPRNG extends EnumeratingPRNG {
  constructor(transcript, shared = null) {
    super(transcript);
    this.shared = shared || {cursor: 0};
  }

  get cursor() {
    return this.shared?.cursor || 0;
  }

  set cursor(value) {
    if (this.shared) this.shared.cursor = value;
  }

  clone() {
    return new ForcedTranscriptPRNG(this.prefix, this.shared);
  }

  decide(key, outcomes) {
    if (outcomes.length === 0) throw new RangeError(`chance request has no outcomes: ${key}`);
    if (outcomes.length === 1) return outcomes[0].value;
    const index = this.cursor;
    this.cursor = index + 1;
    if (index >= this.prefix.length) {
      throw new Error(`candidate replay requested extra chance answer ${index}: ${key}`);
    }
    const selected = this.prefix[index];
    if (selected.key !== key) {
      throw new Error(
        `candidate chance replay diverged at ${index}: expected ${selected.key}, observed ${key}`
      );
    }
    const outcome = outcomes.find(candidate => candidate.choice === selected.choice);
    if (!outcome) throw new Error(`candidate replay choice ${selected.choice} is invalid for ${key}`);
    return outcome.value;
  }
}

// Showdown historically implements secondary-effect percentages by drawing an integer in
// [0,100) and comparing it with a threshold. Expanding all 100 integers is exact but pathological:
// a guaranteed self-drop still draws the unused integer, and two ordinary attacks can create
// 100^n replay prefixes whose outcomes immediately collapse to two categories. During exact
// enumeration only, replace those two call sites with the mathematically identical Bernoulli
// request. Ordinary seeded sessions keep Showdown's original calls and seed consumption.
const originalSelfDrops = BattleActions.prototype.selfDrops;
const originalSecondaries = BattleActions.prototype.secondaries;

BattleActions.prototype.selfDrops = function exactSelfDrops(
  targets, source, move, moveData, isSecondary
) {
  if (!(this.battle.prng instanceof EnumeratingPRNG)) {
    return originalSelfDrops.call(this, targets, source, move, moveData, isSecondary);
  }
  for (const target of targets) {
    if (target === false) continue;
    if (moveData.self && !move.selfDropped) {
      if (!isSecondary && moveData.self.boosts) {
        const applies = typeof moveData.self.chance === "undefined" ||
          this.battle.randomChance(moveData.self.chance, 100);
        if (applies) this.moveHit(source, source, move, moveData.self, isSecondary, true);
        if (!move.multihit) move.selfDropped = true;
      } else {
        this.moveHit(source, source, move, moveData.self, isSecondary, true);
      }
    }
  }
};

BattleActions.prototype.secondaries = function exactSecondaries(
  targets, source, move, moveData, isSelf
) {
  if (!(this.battle.prng instanceof EnumeratingPRNG)) {
    return originalSecondaries.call(this, targets, source, move, moveData, isSelf);
  }
  if (!moveData.secondaries) return;
  for (const target of targets) {
    if (target === false) continue;
    const secondaries = this.battle.runEvent(
      "ModifySecondaries", target, source, moveData, moveData.secondaries.slice()
    );
    for (const secondary of secondaries) {
      const secondaryOverflow = (secondary.boosts || secondary.self) && this.battle.gen <= 8;
      const chance = secondaryOverflow ? secondary.chance % 256 : secondary.chance;
      const applies = typeof secondary.chance === "undefined" ||
        this.battle.randomChance(chance, 100);
      if (applies) this.moveHit(target, source, move, secondary, true, isSelf);
    }
  }
};

function channelLogTail(battle, rootLogLength, channelIds = [0]) {
  const tail = battle.log.slice(rootLogLength).join("\n");
  if (!tail) return Object.fromEntries(channelIds.map(channel => [channel, []]));
  return extractChannelMessages(tail, channelIds);
}

function leafSeed(prefix) {
  const digest = crypto.createHash("sha256").update(JSON.stringify(prefix)).digest("hex");
  return `sodium,${digest}`;
}

function serializeCompletedLeaf(battle, rootLogLength, prefix, includeState) {
  // The enumerator has a common zero seed, so it cannot prevent mechanically identical leaves
  // from merging. Install a deterministic independent future only after the merge key is made.
  const state = battle.toJSON();
  const mergeKey = JSON.stringify(state);
  state.prng = leafSeed(prefix);
  const channels = channelLogTail(battle, rootLogLength, [0, 1, 2]);
  const result = {
    mergeKey,
    state: includeState ? state : undefined,
    ended: Boolean(battle.ended),
    winner: battle.winner || null,
    request_state: battle.requestState,
    turn: battle.turn,
    public_log: channels[0],
    player_logs: {p1: channels[1], p2: channels[2]},
    requests: {
      p1: battle.p1?.activeRequest || null,
      p2: battle.p2?.activeRequest || null,
    },
  };
  return result;
}

function serializeSampledLeaf(battle, rootLogLength, includeState) {
  // A sampled validation path must retain Showdown's real post-turn PRNG state. This makes a
  // sampled multi-turn replay identical to running the same seed through Showdown normally.
  const state = battle.toJSON();
  const channels = channelLogTail(battle, rootLogLength, [0, 1, 2]);
  return {
    state: includeState ? state : undefined,
    ended: Boolean(battle.ended),
    winner: battle.winner || null,
    request_state: battle.requestState,
    turn: battle.turn,
    public_log: channels[0],
    player_logs: {p1: channels[1], p2: channels[2]},
    requests: {
      p1: battle.p1?.activeRequest || null,
      p2: battle.p2?.activeRequest || null,
    },
  };
}

function exactEnumerate(request) {
  const rootText = typeof request.state === "string" ? request.state : JSON.stringify(request.state);
  const choices = request.choices;
  if (!Array.isArray(choices) || choices.length < 2) {
    throw new TypeError("enumerate requires choices=[p1_choice, p2_choice]");
  }

  const minProbability = Number(request.min_probability || 0);
  const maxPaths = Number(request.max_paths || 100000);
  const maxBranches = request.max_branches == null ? null : Number(request.max_branches);
  const includeState = request.include_state !== false;
  if (!(minProbability >= 0 && minProbability < 1)) {
    throw new RangeError("min_probability must be in [0, 1)");
  }
  if (!Number.isSafeInteger(maxPaths) || maxPaths <= 0) {
    throw new RangeError("max_paths must be a positive integer");
  }
  if (maxBranches != null && (!Number.isSafeInteger(maxBranches) || maxBranches <= 0)) {
    throw new RangeError("max_branches must be a positive integer or null");
  }

  let pending = [{prefix: [], probability: fraction(1n)}];
  const completed = new Map();
  let prunedMass = fraction(0n);
  let replayRuns = 0;
  let branchPoints = 0;
  let mergedPrefixes = 0;
  let chanceDepth = 0;
  let maxPending = 1;
  const branchKeyCounts = new Map();

  while (pending.length) {
    chanceDepth++;
    maxPending = Math.max(maxPending, pending.length);
    // Process one chance depth at a time. Distinct replay prefixes can arrive at the identical
    // partially resolved Showdown state (event-order shuffles are the common case). Once the
    // serialized state, unresolved request, and visible log are identical, every continuation is
    // identical too. Add their exact weights and expand only one representative prefix. Waiting
    // until completed leaves to merge allows irrelevant permutations to multiply all later luck.
    const partial = new Map();
    for (const pathState of pending) {
      if (replayRuns >= maxPaths) {
        throw new Error(
          `exact enumeration exceeded max_paths=${maxPaths} after ${branchPoints} branch ` +
          `points, ${mergedPrefixes} merged prefixes, depth ${chanceDepth}, max pending ` +
          `${maxPending}, keys ${JSON.stringify(Object.fromEntries(branchKeyCounts))}; ` +
          `no sampled fallback was used`
        );
      }
      replayRuns++;
      const battle = Battle.fromJSON(rootText);
      battle.restart(() => {});
      const rootLogLength = battle.log.length;
      const replay = new EnumeratingPRNG(pathState.prefix);
      battle.prng = replay;

      try {
        battle.makeChoices(...choices);
        if (replay.cursor !== pathState.prefix.length) {
          throw new Error(
            `chance replay consumed ${replay.cursor} of ${pathState.prefix.length} decisions`
          );
        }
        const leaf = serializeCompletedLeaf(
          battle, rootLogLength, pathState.prefix, includeState
        );
        const existing = completed.get(leaf.mergeKey);
        if (existing) {
          existing.rawProbability = addFractions(existing.rawProbability, pathState.probability);
        } else {
          completed.set(leaf.mergeKey, {...leaf, rawProbability: pathState.probability});
        }
      } catch (error) {
        if (!(error instanceof ChanceBranch)) throw error;
        branchPoints++;
        branchKeyCounts.set(error.key, (branchKeyCounts.get(error.key) || 0) + 1);
        if (error.index !== pathState.prefix.length) {
          throw new Error(
            `branch point ${error.index} does not extend prefix ${pathState.prefix.length}`
          );
        }
        // The exception is raised before the unresolved answer is returned, so the serialized
        // battle represents the state immediately before that request. Include the request's
        // outcome signature because the JS call stack itself is not serialized.
        const partialState = battle.toJSON();
        const outcomeKey = error.outcomes.map(outcome => [
          outcome.choice, outcome.weight.n.toString(), outcome.weight.d.toString(),
        ]);
        const key = JSON.stringify([error.key, outcomeKey, partialState]);
        const existing = partial.get(key);
        if (existing) {
          existing.probability = addFractions(existing.probability, pathState.probability);
          mergedPrefixes++;
        } else {
          partial.set(key, {
            prefix: pathState.prefix,
            probability: pathState.probability,
            key: error.key,
            outcomes: error.outcomes,
          });
        }
      } finally {
        battle.destroy();
      }
    }

    const next = [];
    for (const state of partial.values()) {
      for (const outcome of state.outcomes) {
        const childProbability = multiplyFractions(state.probability, outcome.weight);
        if (fractionNumber(childProbability) < minProbability) {
          prunedMass = addFractions(prunedMass, childProbability);
          continue;
        }
        next.push({
          prefix: [...state.prefix, {key: state.key, choice: outcome.choice}],
          probability: childProbability,
        });
      }
    }
    pending = next;
  }

  let leaves = [...completed.values()];
  leaves.sort((left, right) => {
    const probabilityOrder = compareFractions(right.rawProbability, left.rawProbability);
    return probabilityOrder || left.mergeKey.localeCompare(right.mergeKey);
  });
  if (maxBranches != null && leaves.length > maxBranches) {
    for (const leaf of leaves.slice(maxBranches)) {
      prunedMass = addFractions(prunedMass, leaf.rawProbability);
    }
    leaves = leaves.slice(0, maxBranches);
  }

  let keptMass = fraction(0n);
  for (const leaf of leaves) keptMass = addFractions(keptMass, leaf.rawProbability);
  if (keptMass.n === 0n) throw new Error("chance pruning removed every completed leaf");
  const accountedMass = addFractions(keptMass, prunedMass);
  if (accountedMass.n !== accountedMass.d) {
    throw new Error(
      `chance enumeration accounted for ${accountedMass.n}/${accountedMass.d}, expected 1`
    );
  }

  const responseLeaves = leaves.map(leaf => {
    const normalized = divideFractions(leaf.rawProbability, keptMass);
    return {
      probability: fractionJSON(normalized),
      raw_probability: fractionJSON(leaf.rawProbability),
      state: leaf.state,
      ended: leaf.ended,
      winner: leaf.winner,
      request_state: leaf.request_state,
      turn: leaf.turn,
      public_log: leaf.public_log,
      player_logs: leaf.player_logs,
      requests: leaf.requests,
    };
  });

  return {
    exact: true,
    sampled: false,
    branches: responseLeaves,
    kept_mass: fractionJSON(keptMass),
    pruned_mass: fractionJSON(prunedMass),
    replay_runs: replayRuns,
    branch_points: branchPoints,
    merged_prefixes: mergedPrefixes,
    chance_depth: chanceDepth,
    max_pending: maxPending,
    branch_key_counts: Object.fromEntries(branchKeyCounts),
  };
}

function createState(request) {
  if (!request.options || typeof request.options !== "object") {
    throw new TypeError("create requires Showdown battle options");
  }
  const battle = new Battle(request.options);
  try {
    for (const choices of request.choices || []) {
      if (!Array.isArray(choices)) throw new TypeError("each choice round must be an array");
      battle.makeChoices(...choices);
    }
    return {
      state: battle.toJSON(),
      ended: Boolean(battle.ended),
      winner: battle.winner || null,
      request_state: battle.requestState,
      turn: battle.turn,
    };
  } finally {
    battle.destroy();
  }
}

function sampleOnce(request) {
  const rootText = typeof request.state === "string" ? request.state : JSON.stringify(request.state);
  const battle = Battle.fromJSON(rootText);
  try {
    battle.restart(() => {});
    const rootLogLength = battle.log.length;
    battle.prng = new PRNG(request.seed || null);
    battle.makeChoices(...request.choices);
    return serializeSampledLeaf(battle, rootLogLength, request.include_state !== false);
  } finally {
    battle.destroy();
  }
}

function sessionUpdate(session, includeState = false) {
  const battle = session.battle;
  const newLog = battle.log.slice(session.logPosition).join("\n");
  session.logPosition = battle.log.length;
  const channels = newLog ? extractChannelMessages(newLog, [0, 1, 2]) : {0: [], 1: [], 2: []};
  return {
    messages: {p1: channels[1], p2: channels[2]},
    public_messages: channels[0],
    requests: {
      p1: battle.p1?.activeRequest || null,
      p2: battle.p2?.activeRequest || null,
    },
    state: includeState ? battle.toJSON() : undefined,
    request_state: battle.requestState,
    turn: battle.turn,
    ended: Boolean(battle.ended),
    winner: battle.winner || null,
  };
}

function createSession(request) {
  if (!request.options || typeof request.options !== "object") {
    throw new TypeError("session_create requires Showdown battle options");
  }
  const sessionId = String(request.session_id || crypto.randomUUID());
  if (sessions.has(sessionId)) throw new Error(`session already exists: ${sessionId}`);
  const transcript = [];
  const prng = new RecordingPRNG(new PRNG(request.options.seed || null), transcript);
  const battle = new Battle({...request.options, prng});
  const session = {
    battle,
    logPosition: 0,
    transcript,
    choiceRounds: [],
    options: {...request.options, seed: undefined, prng: undefined},
  };
  sessions.set(sessionId, session);
  return {session_id: sessionId, ...sessionUpdate(session, request.include_state === true)};
}

function normalizeChoice(choice) {
  if (choice == null) return "";
  choice = String(choice).trim();
  if (choice.startsWith("/choose ")) return choice.slice(8);
  if (choice.startsWith("/team ")) return `team ${choice.slice(6)}`;
  return choice;
}

function stepSession(request) {
  const sessionId = String(request.session_id);
  const session = sessions.get(sessionId);
  if (!session) throw new Error(`unknown session: ${sessionId}`);
  if (!Array.isArray(request.choices) || request.choices.length < 2) {
    throw new TypeError("session_step requires choices=[p1_choice, p2_choice]");
  }
  const choices = request.choices.map(normalizeChoice);
  session.battle.makeChoices(...choices);
  session.choiceRounds.push(choices);
  return sessionUpdate(session, request.include_state === true);
}

function normalizedPublicLog(battle) {
  const text = battle.log.join("\n");
  const lines = text ? extractChannelMessages(text, [0])[0] : [];
  return lines.map(line => line.startsWith("|t:|") ? "|t:|" : line);
}

function firstDifference(left, right) {
  const length = Math.max(left.length, right.length);
  for (let i = 0; i < length; i++) {
    if (left[i] !== right[i]) return {index: i, expected: left[i], observed: right[i]};
  }
  return null;
}

function reconstructSessionStrict(request) {
  const sessionId = String(request.session_id);
  const session = sessions.get(sessionId);
  if (!session) throw new Error(`unknown session: ${sessionId}`);
  if (request.p1_team == null || request.p2_team == null) {
    throw new TypeError("session_reconstruct requires p1_team and p2_team");
  }
  const replay = new ForcedTranscriptPRNG(session.transcript);
  const options = {
    ...session.options,
    p1: {...session.options.p1, team: request.p1_team},
    p2: {...session.options.p2, team: request.p2_team},
    prng: replay,
  };
  const battle = new Battle(options);
  try {
    for (const choices of session.choiceRounds) battle.makeChoices(...choices);
    if (replay.cursor !== session.transcript.length) {
      throw new Error(
        `candidate replay consumed ${replay.cursor} of ${session.transcript.length} chance answers`
      );
    }
    const expected = normalizedPublicLog(session.battle);
    const observed = normalizedPublicLog(battle);
    const difference = firstDifference(expected, observed);
    if (difference) {
      throw new Error(
        `candidate public history diverged at line ${difference.index}: ` +
        `expected ${JSON.stringify(difference.expected)}, ` +
        `observed ${JSON.stringify(difference.observed)}`
      );
    }
    return {
      state: battle.toJSON(),
      requests: {p1: battle.p1?.activeRequest || null, p2: battle.p2?.activeRequest || null},
      request_state: battle.requestState,
      turn: battle.turn,
      ended: Boolean(battle.ended),
      winner: battle.winner || null,
      public_log: observed,
      chance_answers: replay.cursor,
    };
  } finally {
    battle.destroy();
  }
}

function getSession(request) {
  const sessionId = String(request.session_id);
  const session = sessions.get(sessionId);
  if (!session) throw new Error(`unknown session: ${sessionId}`);
  return sessionUpdate(session, request.include_state === true);
}

function closeSession(request) {
  const sessionId = String(request.session_id);
  const session = sessions.get(sessionId);
  if (!session) return {closed: false};
  session.battle.destroy();
  sessions.delete(sessionId);
  return {closed: true};
}

function handleRequest(request) {
  switch (request.op) {
  case "ping":
    return {pong: true, showdown_root: showdownRoot};
  case "create":
    return createState(request);
  case "enumerate":
    return exactEnumerate(request);
  case "sample":
    return sampleOnce(request);
  case "session_create":
    return createSession(request);
  case "session_step":
    return stepSession(request);
  case "session_get":
    return getSession(request);
  case "session_reconstruct_strict":
    return reconstructSessionStrict(request);
  case "session_close":
    return closeSession(request);
  default:
    throw new Error(`unknown operation: ${request.op}`);
  }
}

const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
input.on("line", line => {
  if (!line.trim()) return;
  let request;
  try {
    request = JSON.parse(line);
    const result = handleRequest(request);
    process.stdout.write(`${JSON.stringify({id: request.id ?? null, ok: true, result})}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify({
      id: request?.id ?? null,
      ok: false,
      error: String(error?.message || error),
      stack: String(error?.stack || ""),
    })}\n`);
  }
});

input.on("close", () => {
  for (const session of sessions.values()) session.battle.destroy();
  sessions.clear();
});
