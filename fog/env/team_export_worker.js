'use strict';
// Export the validator's parsed sets, so Python need not reimplement paste parsing.
const path = require('path');
const readline = require('readline');
const {Teams, TeamValidator} = require(path.resolve(process.argv[2], 'dist/sim'));
const validator = TeamValidator.get('gen9ou');
readline.createInterface({input: process.stdin}).on('line', line => {
  try {
    const team = Teams.import(JSON.parse(line));
    if (!team || team.length !== 6) throw new Error('Require six submitted Pokemon');
    const problems = validator.validateTeam(team);
    process.stdout.write(JSON.stringify({valid: !problems, problems: problems || [],
      team: problems ? null : team}) + '\n');
  } catch (error) {
    process.stdout.write(JSON.stringify({valid: false, problems: [String(error)], team: null}) + '\n');
  }
});
