import { Tensor } from '../src/tensor';

function runModel(device: 'cpu' | 'gpu') {
    const batch = Tensor.randn([64, 128], false, device);
    const W1 = Tensor.randn([128, 256], true, device);
    const b1 = Tensor.randn([64, 256], true, device);
    const W2 = Tensor.randn([256, 128], true, device);
    const b2 = Tensor.randn([64, 128], true, device);
    const target = Tensor.randn([64, 128], false, device);

    const hidden = batch.matmul(W1).add(b1);
    const output = hidden.matmul(W2).add(b2);
    const diff = output.add(target.mul(-1));
    const sq = diff.mul(diff);
    const loss = sq.mean();
    loss.backward();

    return loss.data;
}

function benchmark(device: 'cpu' | 'gpu', runs = 5) {
    runModel(device);

    const times: number[] = [];
    let lastLoss: any = null;

    for (let i = 0; i < runs; i++) {
        const t0 = performance.now();
        lastLoss = runModel(device);
        const t1 = performance.now();
        times.push(t1 - t0);
    }

    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    return { time: avgTime, loss: lastLoss };
}

const cpuResult = benchmark('cpu');
console.log('CPU avg time:', cpuResult.time.toFixed(2), 'ms');
console.log('CPU loss:', cpuResult.loss);

const gpuResult = benchmark('gpu');
console.log('GPU avg time:', gpuResult.time.toFixed(2), 'ms');
console.log('GPU loss:', gpuResult.loss);
