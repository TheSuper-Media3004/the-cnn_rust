use ndarray::{Array1, Array3};

pub struct Softmax {
    pub weights: ndarray::Array2<f32>,
    pub bias: Array1<f32>,
}

impl Softmax {
    pub fn new(input_size: usize, num_classes: usize) -> Self {
        use ndarray_rand::{rand_distr::Uniform, RandomExt};
        Self {
            weights: ndarray::Array2::random((input_size, num_classes), Uniform::new(-0.1, 0.1)),
            bias: Array1::zeros(num_classes),
        }
    }

    pub fn forward(&self, input: &Array3<f32>, label: usize) -> (Array1<f32>, f32) {
        let input_flat = input.clone().into_raw_vec();
        let input_array = Array1::from(input_flat);
        let logits = self.weights.t().dot(&input_array) + &self.bias;
        let exp = logits.mapv(|x| x.exp());
        let sum_exp = exp.sum();
        let probs = exp.mapv(|x| x / sum_exp);
        let loss = -probs[label].ln();
        (probs, loss)
    }

    pub fn backward(&self, _label: usize) -> Array3<f32> {
        Array3::zeros((1, 1, 1))
    }

    pub fn predict(&self, probs: &Array1<f32>) -> usize {
        probs
            .iter()
            .cloned()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }

    pub fn update(&mut self, _lr: f32) {
        // dummy update
    }
}