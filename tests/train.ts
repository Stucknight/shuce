import { Tensor } from '../src/tensor';

function mseLoss(pred: Tensor, target: Tensor): Tensor {
    const diff = pred.add(target.mul(-1));
    const sq = diff.mul(diff);
    return sq.mean();
}

function updateParams(params: Tensor[], lr: number) {
    for (const p of params) {
        if (p.requires_grad && p.grad) {
            const newData = p.data.map((row, i) =>
                row.map((val, j) => val - lr * p.grad!.data[i][j])
            );
            p.data = newData;
            p.grad = null;
        }
    }
}


let device: 'cpu' | 'gpu' = 'gpu';

const rows = 512;
const inDim = 4196;
const outDim = 512;

let W1 = Tensor.randn([inDim, outDim], true, device);
let X = Tensor.randn([rows, inDim], false, device);
let target = Tensor.randn([rows, outDim], false, device);

const lr = 1e-5;
const steps = 100;

for (let step = 0; step < steps; step++) {
    const Y = X.matmul(W1);
    const loss = mseLoss(Y, target);
    loss.backward();
    console.log(`Step ${step}, Loss:`, loss.data[0][0]);
    updateParams([W1], lr);
}