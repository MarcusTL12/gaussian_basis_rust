use std::collections::HashMap;

use libcint::*;
use ndarray::prelude::*;

const Å2B: f64 = 1.8897261245650618;

fn periodic_table(atom: &str) -> i32 {
    match atom {
        "H" => 1,
        "He" => 2,
        "Li" => 3,
        "Be" => 4,
        "B" => 5,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        "Ne" => 10,
        _ => unimplemented!(),
    }
}

fn parse_atoms(atoms: &str) -> Vec<(String, [f64; 3])> {
    atoms
        .split('\n')
        .filter_map(|l| {
            if l.chars().all(|c| c.is_whitespace()) {
                None
            } else {
                let mut it = l.split_ascii_whitespace();

                let atm = it.next().unwrap().to_owned();
                let coord = [
                    it.next().unwrap().parse().unwrap(),
                    it.next().unwrap().parse().unwrap(),
                    it.next().unwrap().parse().unwrap(),
                ];

                Some((atm, coord))
            }
        })
        .collect()
}

fn normalize_cgto(l: i32, exponents: &[f64], coeffs: &mut ArrayViewMut1<f64>) {
    for (e, c) in exponents.iter().zip(coeffs.iter_mut()) {
        *c *= gto_norm(l, *e);
    }

    let mut self_overlap = 0.0;

    for (&e1, &c1) in exponents.iter().zip(coeffs.iter()) {
        for (&e2, &c2) in exponents.iter().zip(coeffs.iter()) {
            self_overlap += c1 * c2 / gto_norm(l, (e1 + e2) / 2.0).powi(2);
        }
    }

    self_overlap = self_overlap.sqrt();
    coeffs.mapv_inplace(|c| c / self_overlap);
}

fn init_libcint(
    atoms: &Vec<(String, [f64; 3])>,
    basis: &HashMap<String, Vec<(i32, Array2<f64>)>>,
) -> (Vec<[i32; 6]>, Vec<[i32; 8]>, Vec<f64>) {
    let mut atm = Vec::new();
    let mut bas = Vec::new();
    let mut env = vec![0.0; 20];

    for (a, xyz) in atoms {
        atm.push([periodic_table(a), env.len() as i32, 0, 0, 0, 0]);
        env.extend(xyz.iter().map(|x| x * Å2B));
    }

    for (i, (a, _)) in atoms.iter().enumerate() {
        for (l, e_coeffs) in &basis[a] {
            let (h, w) = e_coeffs.dim();
            bas.push([
                i as i32,
                *l,
                h as i32,
                w as i32 - 1,
                0,
                env.len() as i32,
                (env.len() + h) as i32,
                0,
            ]);

            let (exps, coeffs) = e_coeffs.view().split_at(Axis(1), 1);
            let coeffs = coeffs.to_owned();
            let exps_std = exps.as_standard_layout();
            let exps_slice = exps_std.as_slice_memory_order().unwrap();

            env.extend(exps_slice);

            let coeffs = coeffs.t();
            let mut coeffs = coeffs.as_standard_layout();
            for mut r in coeffs.rows_mut() {
                normalize_cgto(*l, exps_slice, &mut r);
            }
            env.extend(coeffs.as_slice_memory_order().unwrap());
        }
    }

    (atm, bas, env)
}

pub struct Molecule {
    atoms: Vec<(String, [f64; 3])>,
    lc_atm: Vec<[i32; 6]>,
    lc_bas: Vec<[i32; 8]>,
    lc_env: Vec<f64>,
}

impl Molecule {
    pub fn get_atoms(&self) -> &[(String, [f64; 3])] {
        &self.atoms
    }

    pub fn new(
        atoms: Vec<(String, [f64; 3])>,
        basis: &HashMap<String, Vec<(i32, Array2<f64>)>>,
    ) -> Self {
        let (lc_atm, lc_bas, lc_env) = init_libcint(&atoms, basis);

        Self {
            atoms,
            lc_atm,
            lc_bas,
            lc_env,
        }
    }

    pub fn from_str(
        atoms: &str,
        basis: &HashMap<String, Vec<(i32, Array2<f64>)>>,
    ) -> Self {
        Self::new(parse_atoms(atoms), basis)
    }

    pub fn int1e<
        'a,
        F: FnMut(
            &'a mut [f64],
            [i32; 2],
            &'a [[i32; 6]],
            &'a [[i32; 8]],
            &'a mut [f64],
        ) -> i32,
    >(
        &'a mut self,
        mut f: F,
        buf: &'a mut [f64],
        shls: [i32; 2],
    ) -> i32 {
        f(buf, shls, &self.lc_atm, &self.lc_bas, &mut self.lc_env)
    }
}
