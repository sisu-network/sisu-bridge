// Auxiliary functions for TS tests
const fs = require('fs');

export const TEST_FILE = 'test_generated.circom';

export const FpFriBits = 64;
export const FpFri = 18446744069414584321n;

export function getTestFile(folder: string) {
  return `./test/${folder}/${TEST_FILE}`;
}

export function writeImportString(folder: string, importString: string) {
  fs.writeFileSync(getTestFile(folder), importString);
  fs.appendFileSync(getTestFile(folder), '\n');
}

export function appendFunctionCall(folder: string, s: string) {
  fs.appendFileSync(getTestFile(folder), s);
}
