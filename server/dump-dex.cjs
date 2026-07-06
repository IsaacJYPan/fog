#!/usr/bin/env node
/**
 * Dump the pinned pokemon-showdown dex to JSON for fog/constants/build.py (spec §1).
 *
 * Usage:
 *   node server/dump-dex.cjs <path-to-pokemon-showdown-checkout> [> dex_dump.json]
 *
 * The checkout must be BUILT first (`node build` inside it) so dist/ exists — we require the compiled
 * dist/sim rather than parsing .ts in Python (see server/README.md). Output goes to stdout as one JSON
 * object: { commit, gen, pokedex, moves, items, abilities, learnsets }.
 */
'use strict';

const path = require('path');
const { execSync } = require('child_process');

const root = path.resolve(process.argv[2] || '.');

let Dex;
try {
  ({ Dex } = require(path.join(root, 'dist', 'sim')));
} catch (err) {
  console.error(
    `Could not require ${path.join(root, 'dist', 'sim')} — did you run \`node build\` in the ` +
    `pokemon-showdown checkout?\n${err.message}`
  );
  process.exit(1);
}

const GEN = 'gen9';
const dex = Dex.mod(GEN).includeData();

let commit = 'unknown';
try {
  commit = execSync('git rev-parse HEAD', { cwd: root, stdio: ['ignore', 'pipe', 'ignore'] })
    .toString()
    .trim();
} catch (_) { /* not a git checkout — leave 'unknown' */ }

const out = { commit, gen: GEN, pokedex: {}, moves: {}, items: {}, abilities: {}, learnsets: {} };

for (const s of dex.species.all()) {
  out.pokedex[s.id] = {
    name: s.name,
    num: s.num,
    types: s.types,
    baseStats: s.baseStats,               // {hp, atk, def, spa, spd, spe}
    abilities: s.abilities,               // {0, 1?, H?, S?} -> names
    weightkg: s.weightkg,
    tier: s.tier || null,
    isNonstandard: s.isNonstandard || null,
    baseSpecies: s.baseSpecies !== s.name ? Dex.toID(s.baseSpecies) : null,
    changesFrom: s.changesFrom ? Dex.toID(s.changesFrom) : null,
    prevo: s.prevo ? Dex.toID(s.prevo) : null,
  };
}

for (const m of dex.moves.all()) {
  out.moves[m.id] = {
    name: m.name,
    num: m.num,
    type: m.type,
    category: m.category,                 // Physical | Special | Status
    basePower: m.basePower,
    accuracy: m.accuracy,                 // number | true (never misses)
    priority: m.priority,
    flags: m.flags,                       // {contact: 1, sound: 1, protect: 1, ...}
    status: m.status || null,
    boosts: m.boosts || null,
    target: m.target,
    selfSwitch: m.selfSwitch || false,
    sideCondition: m.sideCondition || null,
    heal: m.heal || null,
    drain: m.drain || null,
    secondary: m.secondary ? { status: m.secondary.status || null, chance: m.secondary.chance || null } : null,
    isNonstandard: m.isNonstandard || null,
  };
}

for (const it of dex.items.all()) {
  out.items[it.id] = { name: it.name, num: it.num, isNonstandard: it.isNonstandard || null };
}

for (const ab of dex.abilities.all()) {
  out.abilities[ab.id] = { name: ab.name, num: ab.num, isNonstandard: ab.isNonstandard || null };
}

// Learnsets: raw per-species move -> source list ("9L36", "9M", "8L20", ...). Forme/prevo fallback is
// resolved in build.py using baseSpecies/changesFrom/prevo above.
for (const s of dex.species.all()) {
  const entry = dex.data.Learnsets && dex.data.Learnsets[s.id];
  if (entry && entry.learnset) out.learnsets[s.id] = entry.learnset;
}

process.stdout.write(JSON.stringify(out));
