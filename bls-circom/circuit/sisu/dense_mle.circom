pragma circom 2.1.7;

include "./num_func.circom";

// out = left + r * (right - left). This function does not do range checking for input.
template DenseEvaluate_UpdateSinglePoly() {
	signal input left;
	signal input right;
	signal input r;
	signal output out;

	signal tmp <== right - left;
	out <== left + r * tmp;
}

template DenseEvaluate_UpdatePolys(last, poly_len) {
	signal input r;
	signal input poly[poly_len];
	signal output out[poly_len];

	component calculations[poly_len];
	for (var b = 0; b < last; b++) {
		calculations[b] = DenseEvaluate_UpdateSinglePoly();
		calculations[b].left <== poly[b << 1];
		calculations[b].right <== poly[(b << 1) + 1];
		calculations[b].r <== r;

		out[b] <== calculations[b].out;
	}
}

// Evaluate a dense polynomial at a given point. A dense polynomials is the one that has number of
// evaluations close to 2^num_vars.
template DenseEvaluate(eval_len) {
	signal input evals[eval_len];
	var dim = ilog2_ceil(eval_len);
	assert(2**dim == eval_len);
	signal input points[dim];
	signal output out;

	signal reverse_point[dim];
	for (var i = 0; i < dim; i++) {
		reverse_point[i] <== points[dim - 1 - i];
	}

	component update_polys[dim];

	// evaluate single variable of partial point from left to right
	for (var i = 1; i <= dim; i++) {
		var r = reverse_point[i - 1];
		var last = 1 << (dim - i);
		var poly_len = last * 2;
		var index = i - 1;

		update_polys[index] = DenseEvaluate_UpdatePolys(last, poly_len);
		update_polys[index].r <== r;
		if (index == 0) {
			for (var j = 0; j < poly_len; j++) {
				update_polys[index].poly[j] <== evals[j];
			}
		} else {
			for (var j = 0; j < poly_len; j++) {
				update_polys[index].poly[j] <== update_polys[index - 1].out[j];
			}
		}
	}

	out <== update_polys[dim - 1].out[0];
}

template DenseMajorZeroEvaluate(non_zero_group_size, super_group_size, num_groups) {
    var total_evaluations = super_group_size * num_groups;
    var total_num_vars = log2(total_evaluations);

    signal input groups[num_groups][non_zero_group_size];
    signal input points[total_num_vars];

    signal output out;

    // points = [connection_vars zero_vars group_vars]
    var group_num_vars = log2(non_zero_group_size);
    var connection_num_vars = log2(num_groups);
    var zero_num_vars = total_num_vars - connection_num_vars - group_num_vars;

    component group_mle[num_groups];
	for (var i = 0; i < num_groups; i++) {
		group_mle[i] = DenseEvaluate(non_zero_group_size);
		for (var j = 0; j < non_zero_group_size; j++) {
			group_mle[i].evals[j] <== groups[i][j];
		}

		for (var j = 0; j < group_num_vars; j++) {
			group_mle[i].points[j] <== points[connection_num_vars + zero_num_vars + j];
		}
	}

	signal beta_zero[zero_num_vars+1];
	beta_zero[0] <== 1;
	for (var i = 0; i < zero_num_vars; i++) {
		beta_zero[i+1] <== beta_zero[i] * (1 - points[connection_num_vars + i]);
	}

	component final_mle = DenseEvaluate(num_groups);
	for (var i = 0; i < num_groups; i++) {
		final_mle.evals[i] <== group_mle[i].out * beta_zero[zero_num_vars];
	}
	for (var i = 0; i < connection_num_vars; i++) {
		final_mle.points[i] <== points[i];
	}

	out <== final_mle.out;
}

