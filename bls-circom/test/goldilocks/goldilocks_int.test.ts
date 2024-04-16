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
include "../../circuit/sisu/goldilocks_int.circom";
`;
const FOLDER_NAME = "goldilocks";

describe("Goldilock Prime Operations", function() {
  jest.setTimeout(1000 * 1000);

  beforeEach(async function () {
    writeImportString(FOLDER_NAME, IMPORT_STRING);
  });

  it("Goldilock_Add", async function() {
    appendFunctionCall(FOLDER_NAME, `component main = Add(${FpFriBits}, ${FpFri});`);
    const circuit = await wasm_tester(
        path.join(__dirname, '.', TEST_FILE)
    );
    let witness = await circuit.calculateWitness({
      "a": "41345621",
      "b": "789123872381"
    });
    await circuit.assertOut(witness, {"out": "789165218002"});
  });

  it("Goldilock_AddMultiple", async function() {
    appendFunctionCall(FOLDER_NAME, `component main = AddMultiple(${FpFriBits}, ${FpFri}, 4);`);
    const circuit = await wasm_tester(
        path.join(__dirname, '.', TEST_FILE)
    );
    let witness = await circuit.calculateWitness({
      "a": ["1", "2", "3", "4"],
    });
    await circuit.assertOut(witness, {"out": "10"});
  });

  it("Goldilock_Mul", async function() {
    appendFunctionCall(FOLDER_NAME, `component main = Mul(${FpFriBits}, ${FpFri});`);
    const circuit = await wasm_tester(
        path.join(__dirname, '.', TEST_FILE)
    );
    let witness = await circuit.calculateWitness({
      "a": "41345621",
      "b": "789123872381"
    });
    await circuit.assertOut(witness, {"out": "14180072480102609280"});
  });

  it("Goldilock_MulMultipleAdd", async function() {
    const a = [5, 3, 8];
    appendFunctionCall(FOLDER_NAME, `component main = MulMultipleAdd(${FpFriBits}, ${FpFri}, ${a.length});`);
    const circuit = await wasm_tester(
        path.join(__dirname, '.', TEST_FILE)
    );
    let witness = await circuit.calculateWitness({
      "x": "12",
      "y": "36",
      a
    });
    await circuit.assertOut(witness, {"out": "448"});
  });


  it("Goldilock_Square", async function() {
    appendFunctionCall(FOLDER_NAME, `component main = Square(${FpFriBits}, ${FpFri});`);
    const circuit = await wasm_tester(
        path.join(__dirname, '.', TEST_FILE)
    );
    let witness = await circuit.calculateWitness({
      "x": "41345621"
    });
    await circuit.assertOut(witness, {"out": "1709460375875641"});
  });

  it("Goldilock_Cube", async function() {
    appendFunctionCall(FOLDER_NAME, `component main = Cube(${FpFriBits}, ${FpFri});`);
    const circuit = await wasm_tester(
        path.join(__dirname, '.', TEST_FILE)
    );
    let witness = await circuit.calculateWitness({
      "x": "41345621"
    });
    await circuit.assertOut(witness, {"out": "9224285544523384310"});
  });
});
