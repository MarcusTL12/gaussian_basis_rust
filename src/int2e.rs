use arrayvec::ArrayVec;
use itertools::iproduct;
use rayon::prelude::*;

use crate::*;
use matrix_util::*;

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
        matrix: Option<Array5<f64>>,
    ) -> Array5<f64> {
        let n_ao = self.get_n_ao();
        let n_sh = self.get_shells().len();

        let mut matrix = if let Some(m) = matrix {
            assert_eq!(m.dim(), (n_ao, n_ao, n_ao, n_ao, n_comp));
            m
        } else {
            Array5::zeros((n_ao, n_ao, n_ao, n_ao, n_comp))
        };

        let chunks = split_2e_mat_mut(matrix.view_mut(), self.get_shells());

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

                *chunk += &buf_view;
            },
        );

        matrix
    }

    // Very inefficient to use. Only for debug purposes
    pub fn get_int2e_single<
        F: Fn(
            &mut [f64],
            [i32; 4],
            &[[i32; 6]],
            &[[i32; 8]],
            &[f64],
            Option<&mut CINToptimizer>,
        ) -> i32,
    >(
        &self,
        int_func: F,
        n_comp: usize,
        inds: [usize; 4],
        c: usize,
    ) -> f64 {
        let shells = inds
            .iter()
            .map(|&i| self.ao2shell(i))
            .collect::<ArrayVec<_, 4>>()
            .into_inner()
            .unwrap();

        let mut buf = vec![
            0.0;
            shells.iter().map(|s| s.n_ao).product::<usize>()
                * n_comp
        ];

        let shl_inds = shells
            .iter()
            .map(|s| s.shl_ind as i32)
            .collect::<ArrayVec<_, 4>>()
            .into_inner()
            .unwrap();

        self.int(int_func, &mut buf, shl_inds, None);

        let shape = shells
            .iter()
            .map(|s| s.n_ao)
            .chain([n_comp])
            .rev()
            .collect::<ArrayVec<_, 5>>()
            .into_inner()
            .unwrap();

        let buf_view = ArrayView5::from_shape(shape, &buf).unwrap();

        let inds = inds
            .iter()
            .zip(shells.iter())
            .map(|(i, s)| i - s.ao_ind)
            .chain([c])
            .rev()
            .collect::<ArrayVec<_, 5>>()
            .into_inner()
            .unwrap();

        buf_view[inds]
    }
}
