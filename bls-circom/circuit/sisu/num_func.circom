pragma circom 2.1.7;

function dec2bin(d, size) {
    var result[32];
    for (var i = 0; i < size; i++) {
        if (d > 0) {
            result[size - i - 1] = d & 1;
            d = d >> 1;
        } else {
            result[size - i - 1] = 0;
        }
    }

    for (var i = size; i < 32; i++) {
        result[i] = 0;
    }

    return result;
}

function round_to_next_two_pow(n) {
    return two_pow(ilog2_ceil(n));
}

function ilog2_ceil(x) {
  if (x == 0) {
    return x;
  }

  var add = 0;
  var count = 0;
  while (x > 1) {
    if (x % 2 == 1) {
      add = 1;
    }
    x = x \ 2;
    count++;
  }

  return count + add;
}

function div_mod(a, b) {
    var out[2];
    out[0] = a \ b;
    out[1] = a - out[0] * b;
    return out;
}

function log2(n) {
    assert(n > 0);

    var s = 1;
    var result = 0;
    while (1==1) {
        assert(s <= n);
        
        if (s == n) {
            return result;
        }

        s *= 2;
        result += 1;
    }

    return result;
}

function two_pow(n) {
    assert(n < 251); // We only support maximum 251 bit result.

    if (n == 0) {
        return 1;
    }

    if (n == 1) {
        return 2;
    }

    if (n % 2 == 0) {
        var x = two_pow(n/2);
        return x * x;
    } else {
        var x = two_pow((n-1)/2);
        return 2 * x * x;
    }
}
