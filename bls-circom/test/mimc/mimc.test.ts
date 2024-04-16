import path from 'path';

import {
  appendFunctionCall,
  TEST_FILE,
  writeImportString,
} from '../test_utils';

const circom_tester = require('circom_tester');
const wasm_tester = circom_tester.wasm;

const IMPORT_STRING = `pragma circom 2.1.7;
include "../../circuit/sisu/mimc.circom";
`;
const FOLDER_NAME = "mimc";

describe("Mimc", function () {
  jest.setTimeout(1000 * 1000);

  beforeEach(async function () {
    writeImportString(FOLDER_NAME, IMPORT_STRING);
  });

  it("Mimc", async function () {
    appendFunctionCall(FOLDER_NAME, `component main = MimcMultiple(3);`);
    const circuit = await wasm_tester(
      path.join(__dirname, '.', TEST_FILE)
    );
    let witness = await circuit.calculateWitness({
      "in": [123, 96, 200]
    });
    await circuit.assertOut(witness, {
      "out": "10568441763126500804648284785720361326146205341088946821923410731051822484744"
    });
  });
});
