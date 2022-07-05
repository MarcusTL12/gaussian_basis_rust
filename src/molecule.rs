use std::fmt::Debug;

use ao_basis::LazyBasis;
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

pub fn parse_atoms(atoms: &str) -> Vec<(String, [f64; 3])> {
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

#[derive(Clone, Copy, Debug)]
pub struct Shell {
    pub l: usize,
    pub n_cont: usize,
    pub n_ao: usize,
    pub ao_ind: usize,
    pub shl_ind: usize,
}

impl Shell {
    fn new(l: usize, n_cont: usize, ao_ind: usize, shl_ind: usize) -> Self {
        Self {
            l,
            n_cont,
            n_ao: (2 * l + 1) * n_cont,
            ao_ind,
            shl_ind,
        }
    }
}

fn init_libcint(
    atoms: &Vec<(String, [f64; 3])>,
    basis: &mut LazyBasis,
) -> (
    Vec<[i32; 6]>,
    Vec<[i32; 8]>,
    Vec<f64>,
    Vec<Shell>,
    usize,
    usize,
    usize,
) {
    let mut atm = Vec::new();
    let mut bas = Vec::new();
    let mut env = vec![0.0; 20];

    let mut shells = Vec::new();

    let mut nuc_charge = 0;

    for (a, xyz) in atoms {
        let charge = periodic_table(a);
        nuc_charge += charge as usize;
        atm.push([charge, env.len() as i32, 0, 0, 0, 0]);
        env.extend(xyz.iter().map(|x| x * Å2B));
    }

    let mut n_ao = 0;
    let mut max_shell_size = 0;

    for (i, (a, _)) in atoms.iter().enumerate() {
        for (l, e_coeffs) in basis.get(a) {
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

            let shell = Shell::new(*l as usize, w - 1, n_ao, shells.len());
            shells.push(shell);
            n_ao += shell.n_ao;
            if n_ao > max_shell_size {
                max_shell_size = n_ao
            };

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

    (atm, bas, env, shells, n_ao, nuc_charge, max_shell_size)
}

pub struct Molecule {
    atoms: Vec<(String, [f64; 3])>,
    shells: Vec<Shell>,
    lc_atm: Vec<[i32; 6]>,
    lc_bas: Vec<[i32; 8]>,
    lc_env: Vec<f64>,
    n_ao: usize,
    n_el: usize,
    nuc_charge: usize,
    max_shell_size: usize,
}

impl Debug for Molecule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Molecule: {{")?;

        write!(f, "Atoms: [{}", self.atoms[0].0)?;

        for (atom, _) in &self.atoms[1..] {
            write!(f, ", {}", atom)?;
        }

        writeln!(f, "]")?;

        writeln!(f, "n_ao:    {}", self.n_ao)?;
        writeln!(f, "n_el:    {}", self.n_ao)?;
        writeln!(f, "n_ao:    {}", self.n_ao)?;
        writeln!(f, "n_shell: {}", self.shells.len())?;

        writeln!(f, "}}")?;

        Ok(())
    }
}

impl Molecule {
    pub fn new(atoms: Vec<(String, [f64; 3])>, basis: &mut LazyBasis) -> Self {
        let (lc_atm, lc_bas, lc_env, shells, n_ao, nuc_charge, max_shell_size) =
            init_libcint(&atoms, basis);

        Self {
            atoms,
            shells,
            lc_atm,
            lc_bas,
            lc_env,
            n_ao,
            n_el: nuc_charge,
            nuc_charge,
            max_shell_size,
        }
    }

    pub fn get_atoms(&self) -> &[(String, [f64; 3])] {
        &self.atoms
    }

    pub fn get_shell(&self, i: usize) -> Shell {
        self.shells[i]
    }

    pub fn get_shells(&self) -> &[Shell] {
        &self.shells
    }

    pub fn get_n_ao(&self) -> usize {
        self.n_ao
    }

    pub fn get_n_el(&self) -> usize {
        self.n_el
    }

    pub fn get_nuc_charge(&self) -> usize {
        self.nuc_charge
    }

    pub fn set_charge(&mut self, charge: i32) {
        self.n_el = (self.nuc_charge as i32 + charge) as usize;
    }

    pub fn get_max_shell_size(&self) -> usize {
        self.max_shell_size
    }

    pub fn ao2shell(&self, ao_ind: usize) -> Shell {
        *self
            .shells
            .iter()
            .find(|&s| s.ao_ind + s.n_ao > ao_ind)
            .unwrap()
    }

    pub fn int<
        const N_SH: usize,
        F: Fn(
            &mut [f64],
            [i32; N_SH],
            &[[i32; 6]],
            &[[i32; 8]],
            &[f64],
            Option<&mut CINToptimizer>,
        ) -> i32,
    >(
        &self,
        f: F,
        buf: &mut [f64],
        shls: [i32; N_SH],
        opt: Option<&mut CINToptimizer>,
    ) -> i32 {
        f(buf, shls, &self.lc_atm, &self.lc_bas, &self.lc_env, opt)
    }

    pub fn opt<FOpt: Fn(&[[i32; 6]], &[[i32; 8]], &[f64]) -> CINToptimizer>(
        &self,
        f: FOpt,
    ) -> CINToptimizer {
        f(&self.lc_atm, &self.lc_bas, &self.lc_env)
    }

    pub fn get_nuc_rep(&self) -> f64 {
        self.atoms
            .iter()
            .enumerate()
            .flat_map(|(i, atm1)| {
                let c1 = periodic_table(&atm1.0);
                let r1 = atm1.1;
                self.atoms.iter().take(i).map(move |atm2| {
                    let c2 = periodic_table(&atm2.0);
                    let r2 = atm2.1;

                    (c1 * c2) as f64
                        / (r1
                            .iter()
                            .zip(r2.iter())
                            .map(|(x1, x2)| (x1 - x2).powi(2))
                            .sum::<f64>()
                            .sqrt()
                            * Å2B)
                })
            })
            .sum()
    }
}
