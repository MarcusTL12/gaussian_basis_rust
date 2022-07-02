mod basisparse;
mod int1e;
mod int2e;
mod molecule;

pub use basisparse::*;
pub use libcint::*;
pub use molecule::*;
pub use ndarray::*;

#[cfg(test)]
mod tests {
    use std::fs::read_to_string;

    use crate::*;

    fn load_vec(vec_name: &str) -> Vec<f64> {
        read_to_string(format!("./test_data/{}.txt", vec_name))
            .unwrap()
            .split(", ")
            .map(|s| s.parse().unwrap())
            .collect()
    }

    #[test]
    fn test_parse_basis() {
        let bas = get_basis("cc-pvdz");

        assert_eq!(bas["O"].len(), 3);
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
            &get_basis("cc-pvdz"),
        );

        let mut buf = [0.0; 9];

        let ovlp_fn = cint1e!(int1e_ovlp_sph);

        mol.int(ovlp_fn, &mut buf, [0, 0], None);

        // From pyscf
        let check = [
            1.0000000000000002,
            -0.21406265175603023,
            0.19438415205281015,
            -0.21406265175603018,
            1.0000000000000002,
            0.7086073285770356,
            0.19438415205281015,
            0.7086073285770356,
            1.0,
        ];

        for (a, b) in buf.iter().zip(check.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn test_int1e_matrix() {
        let mol = Molecule::new(
            parse_atoms(
                "
    O   0.0     0.0     0.0
    H   1.0     0.0     0.0
    H   0.0     1.0     0.0
",
            ),
            &get_basis("cc-pvdz"),
        );

        let ovlp_fn = cint1e!(int1e_ovlp_sph);

        let ovlp_mat = mol.construct_int1e(ovlp_fn, 1);

        let slice = ovlp_mat.as_slice_memory_order().unwrap();

        // from pyscf
        let check = load_vec("h2o_ovlp_ccpvdz");

        for (a, b) in slice.iter().zip(check.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn test_int2e_matrix() {
        let mol = Molecule::new(
            parse_atoms(
                "
    O   0.0     0.0     0.0
    H   1.0     0.0     0.0
    H   0.0     1.0     0.0
",
            ),
            &get_basis("cc-pvdz"),
        );

        let eri_func = cint2e!(int2e_sph);

        let eri_mat = mol.construct_int2e(eri_func, 1);

        let slice = eri_mat.as_slice_memory_order().unwrap();

        // from pyscf
        let check = load_vec("h2o_eri_ccpvdz");

        for (a, b) in slice.iter().zip(check.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }
}
