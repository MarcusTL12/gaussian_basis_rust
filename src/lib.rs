mod ao_fock;
mod int1e;
mod int2e;
mod matrix_util;
mod molecule;

use ao_basis::{load_basis, LazyBasis};
pub use ao_fock::*;
pub use libcint::*;
pub use molecule::*;
pub use ndarray::*;

pub fn get_basis(basis_name: &str) -> LazyBasis {
    let path = format!("{}/ao_basis/{basis_name}/", env!("OUT_DIR"));
    load_basis(&path)
}

#[cfg(test)]
mod tests {
    const TEST_PREC: f64 = 1e-13;

    use crate::*;

    use std::{
        fs::File,
        io::{BufRead, BufReader},
    };

    use clap::Parser;

    #[derive(Parser, Debug)]
    struct Args {
        #[clap(long)]
        test_threads: Option<usize>,

        #[clap(long, parse(from_flag))]
        show_output: bool,
    }

    fn get_num_test_jobs() -> usize {
        let args = Args::parse();

        args.test_threads.unwrap_or(num_cpus::get())
    }

    fn set_threads() {
        let test_threads = get_num_test_jobs();

        let threads = num_cpus::get() / test_threads;

        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap_or(());
    }

    fn load_vec(vec_name: &str) -> Vec<f64> {
        BufReader::new(
            File::open(format!("test_data/{}.txt", vec_name)).unwrap(),
        )
        .lines()
        .map(|l| l.unwrap().parse().unwrap())
        .collect()
    }

    #[test]
    fn test_parse_basis() {
        let mut bas = get_basis("cc-pvdz");

        assert_eq!(bas.get("O").len(), 5);
    }

    #[test]
    fn test_molecule() {
        let mol = Molecule::new(
            parse_atoms(
                "
    O   0.0     0.0     0.0
    H   1.0     0.0     0.0
    H   0.0     1.0     0.0
",
            ),
            &mut get_basis("cc-pvdz"),
        );

        let mut buf = [0.0; 9];

        let ovlp_fn = cint1e!(int1e_ovlp_sph);

        mol.int(ovlp_fn, &mut buf, [0, 0], None);

        // From pyscf
        let check = [
            1.0000000000000002,
            -0.21406265175603023,
            -0.21406265175603018,
            1.0000000000000002,
        ];

        for (a, b) in buf.iter().zip(check.iter()) {
            assert!((a - b).abs() < TEST_PREC);
        }
    }

    #[test]
    fn test_int1e_matrix() {
        set_threads();

        let mol = Molecule::new(
            parse_atoms(
                "
    O   0.0     0.0     0.0
    H   1.0     0.0     0.0
    H   0.0     1.0     0.0
",
            ),
            &mut get_basis("cc-pvdz"),
        );

        let ovlp_fn = cint1e!(int1e_ovlp_sph);

        let ovlp_mat = mol.construct_int1e(ovlp_fn, 1, None);

        let slice = ovlp_mat.as_slice_memory_order().unwrap();

        // from pyscf
        let check = load_vec("h2o_ovlp_ccpvdz");

        for (a, b) in slice.iter().zip(check.iter()) {
            assert!((a - b).abs() < TEST_PREC);
        }
    }

    #[test]
    fn test_int2e_matrix() {
        set_threads();

        let mol = Molecule::new(
            parse_atoms(
                "
    O   0.0     0.0     0.0
    H   1.0     0.0     0.0
    H   0.0     1.0     0.0
",
            ),
            &mut get_basis("cc-pvdz"),
        );

        let eri_mat = mol.construct_int2e(
            cint2e!(int2e_sph),
            1,
            cint_opt!(int2e_optimizer),
            true,
            None,
        );

        let slice = eri_mat.as_slice_memory_order().unwrap();

        // from pyscf
        let check = load_vec("h2o_eri_ccpvdz");

        assert_eq!(slice.len(), check.len());

        for (a, b) in slice.iter().zip(check.iter()) {
            assert!((a - b).abs() < TEST_PREC);
        }
    }

    #[test]
    fn test_ao_g_construction() {
        set_threads();

        let mol = Molecule::new(
            parse_atoms(
                "
    O   0.0     0.0     0.0
    H   1.0     0.0     0.0
    H   0.0     1.0     0.0
",
            ),
            &mut get_basis("cc-pvdz"),
        );

        let density = load_vec("h2o_rand_D_ccpvdz");
        let density = ArrayView2::from_shape((24, 24), &density).unwrap();

        let ao_g = mol.construct_ao_g(density, None);

        let slice = ao_g.as_slice_memory_order().unwrap();

        let check = load_vec("h2o_rand_G_ccpvdz");

        assert_eq!(slice.len(), check.len());

        for (a, b) in slice.iter().zip(check.iter()) {
            assert!((a - b).abs() < TEST_PREC);
        }
    }

    #[test]
    fn test_h_construction() {
        set_threads();

        let mol = Molecule::new(
            parse_atoms(
                "
    O   0.0     0.0     0.0
    H   1.0     0.0     0.0
    H   0.0     1.0     0.0
",
            ),
            &mut get_basis("cc-pvdz"),
        );

        let h = mol.construct_ao_h(None);

        let slice = h.as_slice_memory_order().unwrap();

        let check = load_vec("h2o_h_ccpvdz");

        assert_eq!(slice.len(), check.len());

        for (a, b) in slice.iter().zip(check.iter()) {
            assert!((a - b).abs() < TEST_PREC);
        }
    }

    #[test]
    fn test_ao_fock_construction() {
        set_threads();

        let mol = Molecule::new(
            parse_atoms(
                "
    O   0.0     0.0     0.0
    H   1.0     0.0     0.0
    H   0.0     1.0     0.0
",
            ),
            &mut get_basis("cc-pvdz"),
        );

        let density = load_vec("h2o_rand_D_ccpvdz");
        let density = ArrayView2::from_shape((24, 24), &density).unwrap();

        let ao_fock = mol.construct_ao_fock(density, None);

        let slice = ao_fock.as_slice_memory_order().unwrap();

        let check = load_vec("h2o_rand_F_ccpvdz");

        assert_eq!(slice.len(), check.len());

        for (a, b) in slice.iter().zip(check.iter()) {
            assert!((a - b).abs() < TEST_PREC);
        }
    }

    #[test]
    fn test_nuc_rep() {
        let mol = Molecule::new(
            parse_atoms(
                "
    O   0.0     0.0     0.0
    H   1.0     0.0     0.0
    H   0.0     1.0     0.0
",
            ),
            &mut get_basis("cc-pvdz"),
        );

        let nuc_rep = mol.get_nuc_rep();

        assert!(nuc_rep - 8.841020169010916 < TEST_PREC);
    }

    #[test]
    fn test_sto_3g() {
        let mol = Molecule::new(
            parse_atoms(
                "
    O   0.0     0.0     0.0
    H   1.0     0.0     0.0
    H   0.0     1.0     0.0
",
            ),
            &mut get_basis("sto-3g"),
        );

        println!("{:?}", mol);
    }
}
