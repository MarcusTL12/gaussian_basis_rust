use crate::*;

pub fn split_1e_mat_mut<'a, I>(
    mut mat: ArrayViewMut<'a, f64, I>,
    shells: &[Shell],
) -> Vec<ArrayViewMut<'a, f64, I>>
where
    I: ndarray::Dimension,
{
    let mut parts = Vec::new();

    for sh1 in shells {
        let (mut top, rest) = mat.split_at(Axis(0), sh1.n_ao);
        mat = rest;

        for sh2 in shells {
            let (chunk, right) = top.split_at(Axis(1), sh2.n_ao);
            top = right;

            parts.push(chunk);
        }
    }

    parts
}

pub fn split_1e_mat<'a, I>(
    mut mat: ArrayView<'a, f64, I>,
    shells: &[Shell],
) -> Vec<ArrayView<'a, f64, I>>
where
    I: ndarray::Dimension,
{
    let mut parts = Vec::new();

    for sh1 in shells {
        let (mut top, rest) = mat.split_at(Axis(0), sh1.n_ao);
        mat = rest;

        for sh2 in shells {
            let (chunk, right) = top.split_at(Axis(1), sh2.n_ao);
            top = right;

            parts.push(chunk);
        }
    }

    parts
}

pub fn split_2e_mat_mut<'a, I>(
    mut mat: ArrayViewMut<'a, f64, I>,
    shells: &[Shell],
) -> Vec<ArrayViewMut<'a, f64, I>>
where
    I: ndarray::Dimension,
{
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
