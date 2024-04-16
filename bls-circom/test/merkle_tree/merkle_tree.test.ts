import path from 'path';

import {
  appendFunctionCall,
  TEST_FILE,
  writeImportString,
} from '../test_utils';

const circom_tester = require('circom_tester');
const wasm_tester = circom_tester.wasm;

const IMPORT_STRING = `pragma circom 2.1.7;
include "../../circuit/sisu/merkle_tree.circom";
`;
const FOLDER_NAME = "merkle_tree";

describe("MerkleTree", function () {
  jest.setTimeout(1000 * 1000);

  beforeEach(async function () {
    writeImportString(FOLDER_NAME, IMPORT_STRING);
  });

  it("VerifyMerklePathMimc", async function () {
    appendFunctionCall(FOLDER_NAME, `component main = VerifyMerklePathMimc(3);`);
    const circuit = await wasm_tester(
      path.join(__dirname, '.', TEST_FILE)
    );
    let witness = await circuit.calculateWitness({
      "value": "3",
      "root": "3206217212261577739334051676958301888321246892951009107947761120159033747844",
      "path": ["4", "13402505180104709716671459243767866689880561571903165814963918592888004261613", "8909211162306377990935799622880054764934132168355377284272537812185000755242"],
      "index": "2",
    });
    await circuit.assertOut(witness, {});
  });

  it("VerifyMultiMerklePathMimc", async function () {
    appendFunctionCall(FOLDER_NAME, `component main = VerifyMultiMerklePathMimc(2, 3);`);
    const circuit = await wasm_tester(
      path.join(__dirname, '.', TEST_FILE)
    );
    let witness = await circuit.calculateWitness({
      "value": "3",
      "root": "3206217212261577739334051676958301888321246892951009107947761120159033747844",
      "path": ["4", "13402505180104709716671459243767866689880561571903165814963918592888004261613", "8909211162306377990935799622880054764934132168355377284272537812185000755242"],
      "index": "2",
    });
    await circuit.assertOut(witness, {});
  });
});
