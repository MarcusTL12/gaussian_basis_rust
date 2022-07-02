use itertools::iproduct;
use rayon::prelude::*;

use crate::*;

fn split_mat<'a>(
    mut mat: ArrayViewMut5<'a, f64>,
    shells: &[Shell],
) -> Vec<ArrayViewMut5<'a, f64>> {
    let mut parts = Vec::new();

    for sh1 in shells {
        let (mut cut1, rest) = mat.split_at(Axis(0), sh1.n_ao);
        mat = rest;
        for sh2 in shells {
            let (mut cut2, rest) = cut1.split_at(Axis(1), sh2.n_ao);
            cut1 = rest;
            for sh3 in shells {
                let (mut cut3, rest) = cut2.split_at(Axis(2), sh3.n_ao);
                cut2 = rest;
                for sh4 in shells {
                    let (chunk, rest) = cut3.split_at(Axis(3), sh4.n_ao);
                    cut3 = rest;

                    parts.push(chunk);
                }
            }
        }
    }

    parts
}

impl Molecule {
    pub fn construct_int2e<
        F: Fn(
                &mut [f64],
                [i32; 4],
                &[[i32; 6]],
                &[[i32; 8]],
                &[f64],
                Option<&mut CINToptimizer>,
            ) -> i32
            + std::marker::Sync,
        FOpt: Fn(&[[i32; 6]], &[[i32; 8]], &[f64]) -> CINToptimizer
            + std::marker::Send
            + std::marker::Sync,
    >(
        &self,
        int_func: F,
        n_comp: usize,
        opt_func: FOpt,
        use_opt: bool,
    ) -> Array5<f64> {
        let n_ao = self.get_n_ao();
        let n_sh = self.get_shells().len();

        let mut matrix = Array5::zeros((n_ao, n_ao, n_ao, n_ao, n_comp));

        let chunks = split_mat(matrix.view_mut(), self.get_shells());

        let mut chunks: Vec<_> = iproduct!(0..n_sh, 0..n_sh, 0..n_sh, 0..n_sh)
            .map(|(i, j, k, l)| [l as i32, k as i32, j as i32, i as i32])
            .zip(chunks.into_iter())
            .collect();

        chunks.par_iter_mut().for_each_init(
            || {
                (
                    vec![0.0; self.get_max_shell_size().pow(4) * n_comp],
                    if use_opt {
                        Some(self.opt(&opt_func))
                    } else {
                        None
                    },
                )
            },
            |(buf, opt), (shls, chunk)| {
                self.int(&int_func, buf, *shls, opt.as_mut());

                let buf_view =
                    ArrayView5::from_shape(chunk.dim(), &buf).unwrap();

                chunk.assign(&buf_view);
            },
        );

        matrix
    }
}
