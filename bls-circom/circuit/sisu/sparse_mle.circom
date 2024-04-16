pragma circom 2.1.7;

include "./num_func.circom";

template PrecomputeEq_SingleIter(i) {
  var len = 1 << i;

  signal input dp[len];
  signal input gi;
  signal output out[2 * len];

  signal mul[len];
  signal sub[len];

  for (var b = 0; b < len; b++) {
    // dp[b + len] = prev * g[i];
    out[b + len] <== dp[b] * gi;

    // dp[b] = prev - dp[b + len];
    out[b] <== dp[b] - out[b + len];
  }
}

/// utility: precompute f(x) = eq(g,x)
template PrecomputeEq(dim) {
  signal input g[dim];
  signal output out[1 << dim];
  component iterations[dim - 1];

  // dp[0] = F::one() - g[0];
  signal dp0 <== 1 - g[0];

  for (var i = 1; i < dim; i++) {
    var len = 1 << i;
    var index = i - 1;

    iterations[index] = PrecomputeEq_SingleIter(i);
    iterations[index].gi <== g[i];
    if (index == 0) {
      iterations[index].dp[0] <== dp0; // dp[0] = F::one() - g[0];
      iterations[index].dp[1] <== g[0]; // dp1 = g0;
    } else {
      for (var j = 0; j < len; j++) {
        iterations[index].dp[j] <== iterations[index - 1].out[j];
      }
    }
  }

  if (dim == 0) {
    out[0] <== dp0;
  } else if (dim == 1) {
    out[0] <== dp0;
    out[1] <== g[0];
  } else {
    for (var i = 0; i < (1 << dim); i++) {
      out[i] <== iterations[dim - 2].out[i];
    }
  }
}

function getOuterLoopCount(eval_len, point_len) {
  var window = ilog2_ceil(eval_len);
  var count = 0;

  while (point_len > 0) {
    count ++;
    var focus_length;
    if (window > 0 && point_len > window) {
      focus_length = window;
    } else {
      focus_length = point_len;
    }

    point_len = point_len - focus_length;
  }

  return count;
}

template UpdateEvalValue() {
  signal input cached_value;
  signal input gz;
  signal input last_eval;
  signal output out;

  out <== cached_value + gz * last_eval;
}

// last_positions is a constant array from Sparse MLE evaluation. This array depends on the
// structure of the gates in the Sisu circuit and does not depend on the gate's values. Hence, they
// could be calculated once a Sisu circuit is finalized.
template SparseMleEvaluate(eval_len, _point_len, flattened_lens, old_idxes, prev_output_indexes, last_positions) {
  // How many "while !points.is_empty()" iteration we have to run?
  var loop_count = getOuterLoopCount(eval_len, _point_len);
  var flattened_len = 0;
  for (var i = 0; i < loop_count; i++) {
    flattened_len += flattened_lens[i];
  }

  signal input points[_point_len];
  signal input evaluations[eval_len];
  signal output out;

  var point_len = _point_len;

  var window = ilog2_ceil(eval_len);
  var inner_index = 0;
  var point_base_index = 0;

  component precomputes[loop_count];
  component updateEvals[flattened_len];
  for (var ii = 0; ii < loop_count; ii++) {
    var focus_length;
    if (window > 0 && point_len > window) {
      focus_length = window;
    } else {
      focus_length = point_len;
    }

    // Precompute
    precomputes[ii] = PrecomputeEq(focus_length);
    for (var i = 0; i < focus_length; i++) {
      precomputes[ii].g[i] <== points[point_base_index + i];
    }

    // Iterate through the last map.
    for (var i = 0; i < flattened_lens[ii]; i++) {
      var old_idx = old_idxes[inner_index];
      var new_idx = old_idx >> focus_length;

      updateEvals[inner_index] = UpdateEvalValue();
      updateEvals[inner_index].gz <== precomputes[ii].out[old_idx & ((1 << focus_length) - 1)];
      if (ii == 0) {
        updateEvals[inner_index].last_eval <== evaluations[i];
      } else {
        updateEvals[inner_index].last_eval <== updateEvals[prev_output_indexes[inner_index - eval_len]].out;
      }
      if (last_positions[inner_index] == 18446744073709551615) { // usize::MAX
        // The value is simply 0.
        updateEvals[inner_index].cached_value <== 0;
      } else {
        // Get this value from the component output.
        updateEvals[inner_index].cached_value <== updateEvals[last_positions[inner_index]].out;
      }

      // log( "cached value = ", updateEvals[inner_index].cached_value,
      //     "last value = ", updateEvals[inner_index].last_eval,
      //     " new value = ", updateEvals[inner_index].out);

      inner_index++;
    }

    point_base_index += focus_length;
    point_len -= focus_length;
  }

  // out <== 0;
  out <== updateEvals[inner_index - 1].out;
}
