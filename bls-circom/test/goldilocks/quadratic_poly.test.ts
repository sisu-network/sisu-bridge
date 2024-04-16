import path from 'path';

import {
  appendFunctionCall,
  FpFri,
  FpFriBits,
  TEST_FILE,
  writeImportString,
} from '../test_utils';

const circom_tester = require('circom_tester');
const wasm_tester = circom_tester.wasm;

const IMPORT_STRING = `pragma circom 2.1.7;
include "../../circuit/sisu/quadratic_poly.circom";
`;
const FOLDER_NAME = "goldilocks";

describe("Quadratic Polynomial", function() {
  jest.setTimeout(1000 * 1000);

  beforeEach(async function () {
    writeImportString(FOLDER_NAME, IMPORT_STRING);
  });

  it("QuadraticPolyEvaluate", async function() {
    appendFunctionCall(FOLDER_NAME, `component main = QuadraticPolyEvaluate(${FpFriBits}, ${FpFri});`);
    const circuit = await wasm_tester(
        path.join(__dirname, '.', TEST_FILE)
    );
    let witness = await circuit.calculateWitness({
      "a0": "3",
      "a1": "4",
      "a2": "5",
      "x": "2"
    });
    await circuit.assertOut(witness, {"out": "31"});
  });

  it("QuadraticPolyHashCoeffs", async function() {
    appendFunctionCall(FOLDER_NAME, `component main = QuadraticPolyHashCoeffs(${FpFriBits}, ${FpFri});`);
    const circuit = await wasm_tester(
        path.join(__dirname, '.', TEST_FILE)
    );
    let witness = await circuit.calculateWitness({
      "a0": "34",
      "a1": "2113",
      "a2": "192"
    });
    await circuit.assertOut(witness, {"out": "11073192571180991937"});
  });
});
