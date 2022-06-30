use std::{collections::HashMap, fs::read_to_string};

use once_cell::sync::Lazy;
use regex::Regex;

use ndarray::Array2;

fn load_basis_file(basis_name: &str) -> String {
    read_to_string(format!("./ao_bases/{}.nw", basis_name)).unwrap()
}

fn get_angular_momentum(c: char) -> i32 {
    match c {
        'S' => 0,
        'P' => 1,
        'D' => 2,
        'F' => 3,
        'G' => 4,
        'H' => 5,
        'I' => 6,
        _ => unimplemented!(),
    }
}

fn parse_basis(basis_file: &str) -> HashMap<String, Vec<(i32, Array2<f64>)>> {
    static REG1: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"BASIS SET: \((.+?)\) -> \[(.+?)\]\n((:?.|\n)+?)(:?#|(:?END))",
        )
        .unwrap()
    });

    static REG2: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"([A-Z][a-z]?) +([A-Z])((?:\n(?: +-?\d+\.\d+(?:E(?:\+|-)\d+)?)+)+)",
        )
        .unwrap()
    });

    static REG3: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"-?\d+\.\d+(?:E(?:\+|-)\d+)?").unwrap());

    REG1.captures_iter(basis_file)
        .map(|c| {
            let mut atom = None;
            let v = REG2
                .captures_iter(&c[3])
                .map(|c| {
                    let mut h = 0;
                    let mut buf: Vec<f64> = Vec::new();

                    for line in c[3].split('\n').skip(1) {
                        h += 1;
                        for c in REG3.captures_iter(line) {
                            buf.push(c[0].parse().unwrap());
                        }
                    }

                    let w = buf.len() / h;

                    if atom.is_none() {
                        atom = Some(c[1].to_owned());
                    }

                    (
                        get_angular_momentum(c[2].chars().next().unwrap()),
                        Array2::from_shape_vec((h, w), buf).unwrap(),
                    )
                })
                .collect();

            (atom.unwrap(), v)
        })
        .collect()
}

pub fn get_basis(basis_name: &str) -> HashMap<String, Vec<(i32, Array2<f64>)>> {
    parse_basis(&load_basis_file(basis_name))
}
