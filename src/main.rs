use mnist::{Mnist, MnistBuilder};
use ndarray::{Array, Array2, Array3, ArrayView2, Axis, s};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

struct Conv2D {
    kernel: Array2<f32>,
}

impl Conv2D {
    fn new(kernel_size: (usize, usize)) -> Self {
        let kernel = Array::random(kernel_size, Uniform::new(-1.0, 1.0));
        Conv2D { kernel }
    }

    fn forward(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        let (h, w) = input.dim();
        let (kh, kw) = self.kernel.dim();
        let oh = h - kh + 1;
        let ow = w - kw + 1;
        let mut output = Array2::<f32>::zeros((oh, ow));

        for i in 0..oh {
            for j in 0..ow {
                let window = input.slice(s![i..i + kh, j..j + kw]);
                output[[i, j]] = (&window * &self.kernel).sum();
            }
        }

        output
    }
}

fn relu(input: &Array2<f32>) -> Array2<f32> {
    input.mapv(|x| if x > 0.0 { x } else { 0.0 })
}

fn max_pool2d(input: &Array2<f32>, pool_size: usize) -> Array2<f32> {
    let (h, w) = input.dim();
    let out_h = h / pool_size;
    let out_w = w / pool_size;
    let mut output = Array2::<f32>::zeros((out_h, out_w));

    for i in 0..out_h {
        for j in 0..out_w {
            let window = input.slice(s![
                i * pool_size..(i + 1) * pool_size,
                j * pool_size..(j + 1) * pool_size
            ]);
            output[[i, j]] = window.iter().fold(f32::MIN, |a, &b| a.max(b));
        }
    }

    output
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp.iter().sum();
    exp.iter().map(|x| x / sum_exp).collect()
}

fn cross_entropy_loss(predicted: &[f32], label: u8) -> f32 {
    -predicted[label as usize].ln()
}

fn main() {
   
    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .base_path("data")
        .label_format_digit()
        .training_set_length(1)
        .download_and_extract() // single example
        .finalize();

    let images = Array3::from_shape_vec((1, 28, 28), trn_img)
        .expect("Reshape failed")
        .map(|x| *x as f32 / 255.0);

    let image0 = images.index_axis(Axis(0), 0);
    let label = trn_lbl[0];

    println!("True Label: {}", label);

    // CNN Forward Pass
    let conv = Conv2D::new((3, 3));
    let conv_out = conv.forward(&image0);
    let relu_out = relu(&conv_out);
    let pooled = max_pool2d(&relu_out, 2);

    // Flatten
    let flat = pooled.iter().cloned().collect::<Vec<f32>>();

    // Dummy dense classifier
    let num_classes = 10;
    let input_len = flat.len();
    let weights = Array::random((num_classes, input_len), Uniform::new(-0.5, 0.5));
    let biases = Array::random(num_classes, Uniform::new(-0.5, 0.5));

    let mut logits = vec![0.0; num_classes];
    for i in 0..num_classes {
        logits[i] = flat
            .iter()
            .zip(weights.row(i))
            .map(|(x, w)| x * w)
            .sum::<f32>() + biases[i];
    }

    let probs = softmax(&logits);
    let loss = cross_entropy_loss(&probs, label);

    println!("Predicted probs: {:?}", probs);
    println!("Cross-entropy loss: {:.4}", loss);
}