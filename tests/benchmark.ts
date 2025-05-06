import { Tensor } from '../src/tensor';

function mseLoss(pred: Tensor, target: Tensor): Tensor {
    const diff = pred.add(target.mul(-1));
    const sq = diff.mul(diff);
    return sq.mean();
}

let device:'cpu' | 'gpu' = 'cpu'

const A = new Tensor([
    new Float32Array([1, 2, 3]),
    new Float32Array([4, 5, 6]),
], true, device);

const B = new Tensor([
    new Float32Array([1, 1]),
    new Float32Array([1, 1]),
], true, device);

const C = new Tensor([
    new Float32Array([0, 1, 0]),
    new Float32Array([1, 0, 1]),
], true, device);

const D = new Tensor([
    new Float32Array([0.5, -1, 2]),
    new Float32Array([1.5, 0, -0.5]),
], true, device);

const E = A.add(C);
const F = E.mul(D);
const D_T = D.transpose();
const G = F.matmul(D_T);
const loss = mseLoss(G, B);
loss.backward();

console.log("=== E = A + C ===");
console.log(E.data);

console.log("=== F = E * D ===");
console.log(F.data);

console.log("=== G = F.matmul(D_T) ===");
console.log(G.data);

console.log("=== Loss ===");
console.log(loss.data);

console.log("=== grad of A ===");
console.log(A.grad?.data);

console.log("=== grad of B ===");
console.log(B.grad?.data);

console.log("=== grad of C ===");
console.log(C.grad?.data);

console.log("=== grad of D ===");
console.log(D.grad?.data);

console.log("=== grad of E ===");
console.log(E.grad?.data);

console.log("=== grad of F ===");
console.log(F.grad?.data);

console.log("=== grad of G ===");
console.log(G.grad?.data);

console.log("=== grad of loss ===");
console.log(loss.grad?.data);
