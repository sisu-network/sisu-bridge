#!/bin/bash

run_all() {
  echo "Running all tests...."
  # Put all the tests you want to run here.
}

# $1 is the circom file name, $2 is the test file name.
echo "Running $2"
cp "$2.json" input.json
../run_test.sh $1 $2
