use std::{
    fs,
    io::{self, Read},
    path::Path,
    str::FromStr,
    time::Instant,
};

use ark_bn254::Bn254;
use ark_ec::pairing::Pairing;
use ark_groth16::ProvingKey;
use ark_serialize::CanonicalDeserialize;
use circom_compat::{CircomBuilder, CircomCircuit};
use color_eyre::Result;
use num_bigint::BigInt;
use serde_json::{Error, Value};
use wtns_file::WtnsFile;

pub trait CircomCustomBuilder<E: Pairing> {
    fn push_inputs_from_json<R: Read>(&mut self, r: &mut R) -> Result<(), Error>;

    fn push_inputs_from_json_file(&mut self, path: &str) -> Result<(), Error> {
        let mut reader = fs::File::open(path).unwrap();
        self.push_inputs_from_json(&mut reader)
    }

    fn build_with_wtns(&mut self, path: &str) -> Result<CircomCircuit<E>>;
}

impl<E: Pairing> CircomCustomBuilder<E> for CircomBuilder<E> {
    fn push_inputs_from_json<R: Read>(&mut self, r: &mut R) -> Result<(), Error> {
        let inputs: Value = serde_json::from_reader(r)?;

        for (name, obj) in inputs.as_object().unwrap().into_iter() {
            handle_input(self, name, obj.clone());
        }

        Ok(())
    }

    fn build_with_wtns(&mut self, path: &str) -> Result<CircomCircuit<E>> {
        let mut circom = self.setup();

        // calculate the witness
        let witness = parse_wtns::<E>(path);
        circom.witness = Some(witness);

        // sanity check
        debug_assert!({
            use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem};
            let cs = ConstraintSystem::<E::ScalarField>::new_ref();
            circom.clone().generate_constraints(cs.clone()).unwrap();
            let is_satisfied = cs.is_satisfied().unwrap();
            if !is_satisfied {
                println!(
                    "Unsatisfied constraint: {:?}",
                    cs.which_is_unsatisfied().unwrap()
                );
            }

            is_satisfied
        });

        Ok(circom)
    }
}

fn handle_input<E: Pairing>(builder: &mut CircomBuilder<E>, name: &str, v: Value) {
    match v {
        Value::Array(values) => {
            for val in values {
                handle_input(builder, name, val)
            }
        }
        Value::Number(n) => {
            let val = n.as_u64().unwrap();
            builder.push_input(name, val);
        }
        Value::String(s) => {
            let val = BigInt::from_str(&s).unwrap();
            builder.push_input(name, val);
        }
        _ => panic!(
            "invalid at field {}: only support type [array, number, string]",
            name
        ),
    }
}

pub fn gen_wtns(path: &str) -> String {
    let input_path = path.to_owned() + ".json";
    let executable_path = path.to_owned();
    println!("use executable file: {}", executable_path);

    fs::create_dir_all("/tmp/sisu/").unwrap();

    let circuit_name = Path::new(&path).file_stem().unwrap();
    let circuit_name = circuit_name.to_str().unwrap();
    let now_time = chrono::offset::Local::now().timestamp().to_string();

    let wtns_path = "/tmp/sisu/".to_owned() + &circuit_name + "_" + &now_time + ".wtns";
    println!("writing to wtns file: {}", wtns_path);

    let mut cmd = std::process::Command::new(executable_path);
    cmd.args([input_path, wtns_path.clone()]);

    let now = Instant::now();
    print!("run executable file...");
    let output = cmd.output().expect("");
    io::Write::write_all(&mut io::stdout(), &output.stdout).unwrap();
    io::Write::write_all(&mut io::stderr(), &output.stderr).unwrap();
    assert!(
        output.status.success(),
        "failed to execute gen witness by C++"
    );
    println!("done {:?}", now.elapsed());

    wtns_path.to_string()
}

pub fn parse_wtns<E: Pairing>(path: &str) -> Vec<E::ScalarField> {
    let wtns = WtnsFile::<32>::read(&mut fs::File::open(path).unwrap()).unwrap();
    let mut result = vec![];

    for f in wtns.witness.0 {
        let b = f.as_bytes();
        result.push(E::ScalarField::deserialize_uncompressed(b).unwrap());
    }

    result
}

pub fn parse_zkey(path: &str) -> ProvingKey<Bn254> {
    circom_compat::read_zkey(&mut fs::File::open(path).unwrap())
        .unwrap()
        .0
}
