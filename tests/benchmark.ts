import { Tensor } from '../src/tensor';

function benchmark(device: 'cpu' | 'gpu') {
    const batch = Tensor.randn([64, 128], false, device);         // input batch
    const W1 = Tensor.randn([128, 256], true, device);            // layer 1 weights
    const b1 = Tensor.randn([64, 256], true, device);             // layer 1 bias
    const W2 = Tensor.randn([256, 128], true, device);            // layer 2 weights
    const b2 = Tensor.randn([64, 128], true, device);             // layer 2 bias
    const target = Tensor.randn([64, 128], false, device);        // target for loss

    const t0 = performance.now();

    const hidden = batch.matmul(W1).add(b1);
    const output = hidden.matmul(W2).add(b2);
    const diff = output.add(target.mul(-1));
    const sq = diff.mul(diff);
    const loss = sq.mean();
    loss.backward();

    const t1 = performance.now();
    return { time: t1 - t0, loss: loss.data };
}

const cpuResult = benchmark('cpu');
console.log('CPU time:', cpuResult.time.toFixed(2), 'ms');
console.log('CPU loss:', cpuResult.loss);

const gpuResult = benchmark('gpu');
console.log('GPU time:', gpuResult.time.toFixed(2), 'ms');
console.log('GPU loss:', gpuResult.loss);
