use libcint::*;
use ndarray::prelude::*;

use super::Molecule;

impl Molecule {
    pub fn construct_int1e<
        F: Fn(
            &mut [f64],
            [i32; 2],
            &[[i32; 6]],
            &[[i32; 8]],
            &[f64],
            Option<&mut CINToptimizer>,
        ) -> i32,
    >(
        &self,
        int_func: F,
        n_comp: usize,
    ) -> Array3<f64> {
        let n_ao = self.get_n_ao();

        let mut matrix = Array3::zeros((n_ao, n_ao, n_comp));

        let mut buf = vec![0.0; self.get_max_shell_size().pow(2) * n_comp];

        for (i, sh) in self.get_shells().iter().enumerate() {
            let i1 = sh.ao_ind;
            let i2 = i1 + sh.n_ao;

            self.int(&int_func, &mut buf, [i as i32; 2], None);

            let buf_view =
                ArrayView3::from_shape((sh.n_ao, sh.n_ao, n_comp), &buf)
                    .unwrap();

            let mut mat_view = matrix.slice_mut(s![i1..i2, i1..i2, ..]);

            mat_view.assign(&buf_view);
        }

        for (i, sh1) in self.get_shells().iter().enumerate() {
            let i1 = sh1.ao_ind;
            let i2 = i1 + sh1.n_ao;
            for (j, sh2) in self.get_shells().iter().take(i).enumerate() {
                let j1 = sh2.ao_ind;
                let j2 = j1 + sh2.n_ao;

                self.int(&int_func, &mut buf, [j as i32, i as i32], None);

                let mut buf_view =
                    ArrayView3::from_shape((sh1.n_ao, sh2.n_ao, n_comp), &buf)
                        .unwrap();

                let mut mat_view = matrix.slice_mut(s![i1..i2, j1..j2, ..]);

                mat_view.assign(&buf_view);

                let mut mat_view = matrix.slice_mut(s![j1..j2, i1..i2, ..]);

                buf_view.swap_axes(0, 1);

                mat_view.assign(&buf_view);
            }
        }

        matrix
    }
}
