use libcint::*;
use ndarray::prelude::*;

use super::Molecule;

impl Molecule {
    pub fn construct_int1e<
        'a,
        F: FnMut(
            &mut [f64],
            [i32; 2],
            &'a [[i32; 6]],
            &'a [[i32; 8]],
            &'a [f64],
            Option<&CINToptimizer>,
        ) -> i32,
    >(
        &'a mut self,
        mut int_func: F,
        n_comp: usize,
    ) -> Array3<f64> {
        let n_ao = self.get_n_ao();

        let mut matrix = Array3::zeros((n_ao, n_ao, n_comp));

        let mut buf = vec![0.0; self.get_max_shell_size().pow(2) * n_comp];

        // self.int(int_func, &mut buf, [0 as i32; 2], None);

        for i in 0..self.get_shells().len() {
            let sh = self.get_shell(i);
            let i1 = sh.ao_ind;
            let i2 = i1 + sh.n_ao;

            // self.int(int_func, &mut buf, [i as i32; 2], None);

            let mat_view = matrix.slice(s![i1..i2, i1..i2, ..]);
        }

        for (i, sh1) in self.get_shells().iter().enumerate() {
            for sh2 in self.get_shells().iter().take(i) {}
        }

        matrix
    }
}
