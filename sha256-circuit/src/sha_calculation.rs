use std::collections::HashMap;

use crate::constants::K_VALUES;

use sisulib::common::dec2bin;
pub struct Sha256Cal;

impl Sha256Cal {}

impl Sha256Cal {
    pub fn xor(xs: &[Vec<i64>]) -> Vec<i64> {
        let mut res = vec![0i64; xs[0].len()];
        for i in 0..xs[0].len() {
            for j in 0..xs.len() {
                if j == 0 {
                    res[i] = xs[0][i];
                } else {
                    res[i] = res[i] ^ xs[j][i];
                }
            }
        }

        return res;
    }

    pub fn and(x: &[i64], y: &[i64]) -> Vec<i64> {
        let mut z: Vec<i64> = vec![];
        for (i, j) in x.iter().zip(y) {
            z.push(i & j);
        }

        return z;
    }

    pub fn convert_to_bits(v: u8, bit_length: usize) -> Vec<i64> {
        let mut x = v;
        let mut res: Vec<i64> = vec![];
        for _ in 0..bit_length {
            res.push((x as i64) & 1);
            x /= 2;
        }
        res.reverse();

        return res.to_vec();
    }

    pub fn sum_up(xs: &[&Vec<i64>]) -> Vec<i64> {
        let mut carry = 0i64;
        let mut res: Vec<i64> = vec![];
        for i in (0..xs[0].len()).rev() {
            let mut sum = carry;
            for x in xs {
                sum += x[i];
            }

            carry = sum / 2;
            sum = sum & 1;

            res.push(sum);
        }
        res.reverse();

        return res;
    }

    pub fn sum_up2(xs: &[&Vec<i64>], min_bit_length: usize) -> Vec<i64> {
        let mut carry = 0i64;
        let mut res: Vec<i64> = vec![];
        for i in (0..xs[0].len()).rev() {
            let mut sum = carry;
            for x in xs {
                sum += x[i];
            }

            carry = sum / 2;
            sum = sum & 1;

            res.push(sum);
        }

        while carry > 0 {
            res.push(carry & 1);
            carry >>= 1;
        }

        let diff = min_bit_length - res.len();
        if diff > 0 {
            for _ in 0..diff {
                res.push(0);
            }
        }

        res.reverse();
        return res;
    }

    pub fn right_rotate(x: &[i64], len: usize) -> Vec<i64> {
        let mut res: Vec<i64> = vec![];
        res.extend_from_slice(&x[x.len() - len..]);
        res.extend_from_slice(&x[..x.len() - len]);
        return res;
    }

    pub fn small_sigma0(x: &[i64]) -> Vec<i64> {
        let mut xs: Vec<Vec<i64>> = vec![];

        let rotr7 = Sha256Cal::right_rotate(x, 7);
        let rotr18 = Sha256Cal::right_rotate(x, 18);
        let mut shr3 = vec![0, 0, 0];
        shr3.extend_from_slice(&x[..x.len() - 3]);

        xs.push(rotr7);
        xs.push(rotr18);
        xs.push(shr3);

        return Sha256Cal::xor(xs.as_slice());
    }

    pub fn small_sigma1(x: &[i64]) -> Vec<i64> {
        let mut xs: Vec<Vec<i64>> = vec![];

        let rotr17 = Sha256Cal::right_rotate(x, 17);
        let rotr18 = Sha256Cal::right_rotate(x, 19);
        let mut shr10 = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        shr10.extend_from_slice(&x[..x.len() - 10]);

        xs.push(rotr17);
        xs.push(rotr18);
        xs.push(shr10);

        return Sha256Cal::xor(xs.as_slice());
    }

    pub fn big_sigma0(x: &[i64]) -> Vec<i64> {
        let mut xs: Vec<Vec<i64>> = vec![];

        let rotr2 = Sha256Cal::right_rotate(x, 2);
        let rotr13 = Sha256Cal::right_rotate(x, 13);
        let rotr22 = Sha256Cal::right_rotate(x, 22);

        xs.push(rotr2);
        xs.push(rotr13);
        xs.push(rotr22);

        return Sha256Cal::xor(xs.as_slice());
    }

    pub fn big_sigma1(x: &[i64]) -> Vec<i64> {
        let mut xs: Vec<Vec<i64>> = vec![];

        let rotr6 = Sha256Cal::right_rotate(x, 6);
        let rotr11 = Sha256Cal::right_rotate(x, 11);
        let rotr25 = Sha256Cal::right_rotate(x, 25);

        xs.push(rotr6);
        xs.push(rotr11);
        xs.push(rotr25);

        return Sha256Cal::xor(xs.as_slice());
    }

    pub fn not_and(x: &[i64], y: &[i64]) -> Vec<i64> {
        let mut res: Vec<i64> = vec![];
        for (i, j) in x.iter().zip(y) {
            res.push(!i & j);
        }

        return res;
    }

    pub fn ch(e: &[i64], f: &[i64], g: &[i64]) -> Vec<i64> {
        let tmp1 = Sha256Cal::and(e, f);
        let tmp2 = Sha256Cal::not_and(e, g);

        return Sha256Cal::xor(&vec![tmp1, tmp2]);
    }

    pub fn maj(a: &[i64], b: &[i64], c: &[i64]) -> Vec<i64> {
        let ab = Sha256Cal::and(a, b);
        let ac = Sha256Cal::and(a, c);
        let bc = Sha256Cal::and(b, c);

        return Sha256Cal::xor(&vec![ab, ac, bc]);
    }

    pub fn preprocess_message(m: &[u8]) -> Vec<u8> {
        let length = m.len() * 8;
        let mut message = m.to_vec();

        message.push(0x80);

        while (message.len() * 8 + 64) % 512 != 0 {
            message.push(0x00);
        }

        message.extend(length.to_be_bytes());
        assert!(message.len() * 8 % 512 == 0);

        message
    }

    pub fn split_w(m: &[u8]) -> Vec<Vec<i64>> {
        assert_eq!(m.len(), 64); // 64 bytes == 16 words.

        let mut w: Vec<Vec<i64>> = vec![];
        for i in 0..16 {
            let mut x: Vec<i64> = vec![];
            for j in 0..4 {
                let y = Sha256Cal::convert_to_bits(m[i * 4 + j], 8);
                x.extend(&y);
            }

            w.push(x);
        }

        return w;
    }

    pub fn calculate(init_h: &[u32], m: &[u8]) -> (Vec<u8>, HashMap<String, Vec<Vec<i64>>>) {
        let mut w = Sha256Cal::split_w(m);
        let mut w_i_bits: Vec<Vec<i64>> = Vec::with_capacity(64);
        for i in 0..16 {
            w_i_bits.push(vec![]);
            w_i_bits[i].extend_from_slice(w[i].as_slice());
            w_i_bits[i].extend_from_slice(&[0, 0]);
        }

        for i in 16..64 {
            let s0 = Sha256Cal::small_sigma0(&w[i - 15]);
            let s1 = Sha256Cal::small_sigma1(&w[i - 2]);
            w.push(Sha256Cal::sum_up(&[&w[i - 16], &s0, &w[i - 7], &s1]));

            let x = Sha256Cal::sum_up2(&[&w[i - 16], &s0, &w[i - 7], &s1], 34);
            w_i_bits.push(vec![]);
            w_i_bits[i].extend_from_slice(&x[2..]);
            w_i_bits[i].extend_from_slice(&x[..2]);
        }

        let mut k_i_bits: Vec<Vec<i64>> = Vec::with_capacity(64);
        for i in 0..64 {
            k_i_bits.push(dec2bin(K_VALUES[i] as u64, 32));
        }
        let mut h_i_bits: Vec<Vec<i64>> = Vec::with_capacity(8);
        for i in 0..8 {
            h_i_bits.push(dec2bin(init_h[i] as u64, 32));
        }

        let mut a = h_i_bits[0].clone();
        let mut b = h_i_bits[1].clone();
        let mut c = h_i_bits[2].clone();
        let mut d = h_i_bits[3].clone();
        let mut e = h_i_bits[4].clone();
        let mut f = h_i_bits[5].clone();
        let mut g = h_i_bits[6].clone();
        let mut h = h_i_bits[7].clone();

        // Temp values
        let mut a_i_bits: Vec<Vec<i64>> = Vec::with_capacity(64);
        let mut e_i_bits: Vec<Vec<i64>> = Vec::with_capacity(64);

        for i in 0..64 {
            let big_sigma0 = Sha256Cal::big_sigma0(&a);
            let big_sigma1 = Sha256Cal::big_sigma1(&e);
            let ch = Sha256Cal::ch(&e, &f, &g);
            let temp1 = Sha256Cal::sum_up(&[&h, &big_sigma1, &ch, &k_i_bits[i], &w[i]]);
            let maj = Sha256Cal::maj(&a, &b, &c);
            let temp2 = Sha256Cal::sum_up(&[&big_sigma0, &maj]);

            let x_a = Sha256Cal::sum_up2(
                &[&h, &big_sigma1, &ch, &k_i_bits[i], &w[i], &big_sigma0, &maj],
                35,
            );
            a_i_bits.push(vec![]);
            a_i_bits[i].extend_from_slice(&x_a[3..]);
            a_i_bits[i].extend_from_slice(&x_a[..3]);

            let x_e = Sha256Cal::sum_up2(&[&d, &h, &big_sigma1, &ch, &k_i_bits[i], &w[i]], 35);
            e_i_bits.push(vec![]);
            e_i_bits[i].extend_from_slice(&x_e[3..]);
            e_i_bits[i].extend_from_slice(&x_e[..3]);

            h = g.clone();
            g = f.clone();
            f = e.clone();
            e = Sha256Cal::sum_up(&[&d, &temp1]);
            d = c;
            c = b;
            b = a;
            a = Sha256Cal::sum_up(&[&temp1, &temp2]);
        }

        let mut hout_i_bits: Vec<Vec<i64>> = vec![];
        hout_i_bits.extend_from_slice(&[
            Sha256Cal::sum_up2(&[&h_i_bits[0], &a], 33),
            Sha256Cal::sum_up2(&[&h_i_bits[1], &b], 33),
            Sha256Cal::sum_up2(&[&h_i_bits[2], &c], 33),
            Sha256Cal::sum_up2(&[&h_i_bits[3], &d], 33),
            Sha256Cal::sum_up2(&[&h_i_bits[4], &e], 33),
            Sha256Cal::sum_up2(&[&h_i_bits[5], &f], 33),
            Sha256Cal::sum_up2(&[&h_i_bits[6], &g], 33),
            Sha256Cal::sum_up2(&[&h_i_bits[7], &h], 33),
        ]);

        let mut hash = vec![];
        for i in 0..8 {
            let first_bit = hout_i_bits[i][0];

            hout_i_bits[i] = hout_i_bits[i][1..].to_vec();
            hout_i_bits[i].push(first_bit);

            for x in 0..4 {
                let mut b = 0;
                for y in 0..8 {
                    assert!(hout_i_bits[i][x * 8 + y] == 0 || hout_i_bits[i][x * 8 + y] == 1);
                    b = (b << 1) | (hout_i_bits[i][x * 8 + y] as u8);
                }
                hash.push(b)
            }
        }

        let mut tmp_values = HashMap::<String, Vec<Vec<i64>>>::default();
        tmp_values.insert("a_i_bits".to_string(), a_i_bits);
        tmp_values.insert("e_i_bits".to_string(), e_i_bits);
        tmp_values.insert("w_i_bits".to_string(), w_i_bits);
        tmp_values.insert("h_i_bits".to_string(), h_i_bits);
        tmp_values.insert("k_i_bits".to_string(), k_i_bits);
        tmp_values.insert("hout_i_bits".to_string(), hout_i_bits);

        (hash, tmp_values)
    }
}

#[cfg(test)]
mod tests {
    use sha2::Sha256;

    use crate::constants::INIT_H_VALUES;

    use super::Sha256Cal;
    use sha2::Digest;

    #[test]
    fn calculate_w() {
        let a = "Hello".as_bytes();
        let msg = Sha256Cal::preprocess_message(a);
        let (out, _) = Sha256Cal::calculate(&INIT_H_VALUES, &msg);

        let mut sha_hasher = Sha256::default();
        sha_hasher.update(&a);

        println!("out: {:?}", out);
        println!("sha: {:?}", sha_hasher.finalize());
    }
}
