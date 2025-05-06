import { GPU } from 'gpu.js';

const gpu = new GPU();

function shapeOf(data: Float32Array[] | number[][]): number[] {
    return [data.length, data[0].length];
}

function fillLike(shape: number[], value: number): Float32Array[] {
    const out: Float32Array[] = [];
    for (let i = 0; i < shape[0]; i++) {
        const row = new Float32Array(shape[1]);
        row.fill(value);
        out.push(row);
    }
    return out;
}

function addRows(a: Float32Array[], b: Float32Array[]): Float32Array[] {
    return a.map((row, i) => row.map((val, j) => val + b[i][j]));
}

function mulRows(a: Float32Array[], b: Float32Array[]): Float32Array[] {
    return a.map((row, i) => row.map((val, j) => val * b[i][j]));
}

function dot(a: Float32Array[], b: Float32Array[]): Float32Array[] {
    const m = a.length, k = a[0].length, n = b[0].length;
    const out: Float32Array[] = [];
    for (let i = 0; i < m; i++) {
        out[i] = new Float32Array(n);
    }
    for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++)
            for (let x = 0; x < k; x++)
                out[i][j] += a[i][x] * b[x][j];
    return out;
}

export class Tensor {
    grad: Tensor | null = null;
    op: any = null;
    parents: Tensor[] = [];

    constructor(
        public data: Float32Array[],
        public requires_grad = false,
        public device: 'cpu' | 'gpu' = 'cpu',
        public shape = shapeOf(data),
    ) {}

    static randn(shape: number[], requires_grad = false, device: 'cpu' | 'gpu' = 'cpu'): Tensor {
        const [h, w] = shape;
        const rows: Float32Array[] = [];
        for (let i = 0; i < h; i++) {
            const row = new Float32Array(w);
            for (let j = 0; j < w; j++) {
                const u1 = Math.random(), u2 = Math.random();
                const r = Math.sqrt(-2 * Math.log(u1)), theta = 2 * Math.PI * u2;
                row[j] = r * Math.cos(theta);
            }
            rows.push(row);
        }
        return new Tensor(rows, requires_grad, device, shape);
    }

    add(other: Tensor | number): Tensor {
        return new Add().forward(this, other);
    }

    mul(other: Tensor | number): Tensor {
        return new Mul().forward(this, other);
    }

    matmul(other: Tensor): Tensor {
        return new MatMul().forward(this, other);
    }

    mean(): Tensor {
        return new Mean().forward(this);
    }

    transpose(): Tensor {
        return new Transpose().forward(this);
    }

    static transposeRaw(data: Float32Array[]): Float32Array[] {
        const rows = data.length, cols = data[0].length;
        const out: Float32Array[] = [];
        for (let j = 0; j < cols; j++) {
            out[j] = new Float32Array(rows);
        }
        for (let i = 0; i < rows; i++)
            for (let j = 0; j < cols; j++)
                out[j][i] = data[i][j];
        return out;
    }

    backward(grad: Tensor | null = null): void {
        if (!this.requires_grad) return;
        if (!grad) grad = new Tensor(fillLike(this.shape, 1));
        this.grad ??= new Tensor(fillLike(this.shape, 0));
        this.grad.data = addRows(this.grad.data, grad.data);
        this.op?.backward(grad, this);
    }
}

class Add {
    private kernel = gpu.createKernel(function (a: number[][], b: number[][]) {
        return a[this.thread.y][this.thread.x] + b[this.thread.y][this.thread.x];
    }).setDynamicOutput(true).setDynamicArguments(true);

    forward(a: Tensor, b: Tensor | number): Tensor {
        const shape = a.shape;
        const bTensor = typeof b === 'number' ? new Tensor(fillLike(shape, b), false, a.device, shape) : b as Tensor;
        const outData = a.device === 'gpu'
            ? this.kernel.setOutput([shape[1], shape[0]])(a.data, bTensor.data) as Float32Array[]
            : addRows(a.data, bTensor.data);
        const out = new Tensor(outData, a.requires_grad || bTensor.requires_grad, a.device, shape);
        out.op = this;
        out.parents = [a, bTensor];
        return out;
    }

    backward(grad: Tensor, out: Tensor) {
        const [a, b] = out.parents;
        if (a.requires_grad) a.backward(grad);
        if (b.requires_grad) b.backward(grad);
    }
}

class Mul {
    private kernel = gpu.createKernel(function (a: number[][], b: number[][]) {
        return a[this.thread.y][this.thread.x] * b[this.thread.y][this.thread.x];
    }).setDynamicOutput(true).setDynamicArguments(true);

    forward(a: Tensor, b: Tensor | number): Tensor {
        const shape = a.shape;
        const bTensor = typeof b === 'number' ? new Tensor(fillLike(shape, b), false, a.device, shape) : b as Tensor;
        const outData = a.device === 'gpu'
            ? this.kernel.setOutput([shape[1], shape[0]])(a.data, bTensor.data) as Float32Array[]
            : mulRows(a.data, bTensor.data);
        const out = new Tensor(outData, a.requires_grad || bTensor.requires_grad, a.device, shape);
        out.op = this;
        out.parents = [a, bTensor];
        return out;
    }

    backward(grad: Tensor, out: Tensor) {
        const [a, b] = out.parents;
        if (a.requires_grad) {
            const dA = mulRows(grad.data, b.data);
            a.backward(new Tensor(dA, false, a.device, a.shape));
        }
        if (b.requires_grad) {
            const dB = mulRows(grad.data, a.data);
            b.backward(new Tensor(dB, false, b.device, b.shape));
        }
    }
}

class MatMul {
    private forwardKernel = gpu.createKernel(function (a: number[][], b: number[][], k: number) {
        let sum = 0;
        for (let i = 0; i < k; i++) {
            sum += a[this.thread.y][i] * b[i][this.thread.x];
        }
        return sum;
    }).setDynamicOutput(true).setDynamicArguments(true);

    private backwardAKernel = gpu.createKernel(function (grad: number[][], bT: number[][], n: number) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
            sum += grad[this.thread.y][i] * bT[i][this.thread.x];
        }
        return sum;
    }).setDynamicOutput(true).setDynamicArguments(true);

    private backwardBKernel = gpu.createKernel(function (aT: number[][], grad: number[][], m: number) {
        let sum = 0;
        for (let i = 0; i < m; i++) {
            sum += aT[this.thread.x][i] * grad[i][this.thread.y];
        }
        return sum;
    }).setDynamicOutput(true).setDynamicArguments(true);

    forward(a: Tensor, b: Tensor): Tensor {
        const [m, k1] = a.shape, [k2, n] = b.shape;
        if (k1 !== k2) throw new Error("Shape mismatch for matmul");

        const result = a.device === 'gpu'
            ? this.forwardKernel.setOutput([n, m])(a.data, b.data, k1) as Float32Array[]
            : dot(a.data, b.data);

        const out = new Tensor(result, a.requires_grad || b.requires_grad, a.device, [m, n]);
        out.op = this;
        out.parents = [a, b];
        return out;
    }

    backward(grad: Tensor, out: Tensor): void {
        const [a, b] = out.parents;
        const [m, k] = a.shape, [_, n] = b.shape;

        if (a.requires_grad) {
            const bT = Tensor.transposeRaw(b.data);
            const dA = a.device === 'gpu'
                ? this.backwardAKernel.setOutput([k, m])(grad.data, bT, n) as Float32Array[]
                : dot(grad.data, bT);
            a.backward(new Tensor(dA, false, a.device, [m, k]));
        }

        if (b.requires_grad) {
            const aT = Tensor.transposeRaw(a.data);
            const dB = b.device === 'gpu'
                ? this.backwardBKernel.setOutput([n, k])(aT, grad.data, m) as Float32Array[]
                : dot(aT, grad.data);
            b.backward(new Tensor(dB, false, b.device, [k, n]));
        }
    }
}

class Mean {
    count = 1;

    forward(x: Tensor): Tensor {
        const flat: number[] = [];
        for (const row of x.data) for (let j = 0; j < row.length; j++) flat.push(row[j]);
        this.count = flat.length;
        const sum = flat.reduce((acc, v) => acc + v, 0);
        const out = new Tensor([new Float32Array([sum / this.count])], x.requires_grad, x.device, [1, 1]);
        out.op = this;
        out.parents = [x];
        return out;
    }

    backward(grad: Tensor, out: Tensor) {
        const [x] = out.parents;
        const g = grad.data[0][0] / this.count;
        x.backward(new Tensor(fillLike(x.shape, g)));
    }
}

class Transpose {
    forward(t: Tensor): Tensor {
        const transposed = Tensor.transposeRaw(t.data);
        const out = new Tensor(transposed, t.requires_grad, t.device, [t.shape[1], t.shape[0]]);
        out.op = this;
        out.parents = [t];
        return out;
    }

    backward(grad: Tensor, out: Tensor): void {
        const [t] = out.parents;
        const transposedGrad = Tensor.transposeRaw(grad.data);
        t.backward(new Tensor(transposedGrad, false, t.device, [out.shape[1], out.shape[0]]));
    }
}
