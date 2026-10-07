'use strict';
// Batched validation through the pinned Showdown implementation.
const path = require('path');
const readline = require('readline');
const {Teams, TeamValidator} = require(path.resolve(process.argv[2], 'dist/sim'));
const validator = TeamValidator.get(process.argv[3] || 'gen9ou');
readline.createInterface({input: process.stdin}).on('line', line => {
  try {
    const paste = JSON.parse(line);
    const team = Teams.import(paste);
    const problems = validator.validateTeam(team);
    process.stdout.write(JSON.stringify({valid: !problems, problems: problems || [],
      packed: !problems ? Teams.pack(team) : null}) + '\n');
  } catch (error) {
    process.stdout.write(JSON.stringify({valid: false, problems: [String(error)]}) + '\n');
  }
});
