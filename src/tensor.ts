import {GPU} from 'gpu.js';

const gpu = new GPU();

function deepCopy<T>(x: T): T {
    return Array.isArray(x) ? (x as any[]).map(deepCopy) as any : x;
}

function toNestedArray(data: any): number[][] {
    return Array.isArray(data) ? data.map((row: any) => Array.from(row)) : [];
}

function shapeOf(data: any): number[] {
    const shape: number[] = [];
    let current = data;
    while (Array.isArray(current)) {
        shape.push(current.length);
        current = current[0];
    }
    return shape;
}

function flatDeep(arr: any): number[] {
    return Array.isArray(arr) ? arr.flat(Infinity) : [arr];
}

function reshape(flat: number[], shape: number[]): any {
    if (shape.length === 0) return flat[0];
    const [dim, ...rest] = shape;
    const step = rest.reduce((a, b) => a * b, 1);
    return Array.from({length: dim}, (_, i) =>
        reshape(flat.slice(i * step, (i + 1) * step), rest)
    );
}

function fillLike(template: any, value: number): any {
    return Array.isArray(template)
        ? template.map(row => fillLike(row, value))
        : value;
}

function addArrays(a: any, b: any): any {
    if (!Array.isArray(a) && !Array.isArray(b)) return a + b;
    if (!Array.isArray(a)) return addArrays(fillLike(b, a), b);
    if (!Array.isArray(b)) return addArrays(a, fillLike(a, b));
    return a.map((v, i) => addArrays(v, b[i]));
}

function mulArrays(a: any, b: any): any {
    if (!Array.isArray(a) && !Array.isArray(b)) return a * b;
    if (!Array.isArray(a)) return mulArrays(fillLike(b, a), b);
    if (!Array.isArray(b)) return mulArrays(a, fillLike(a, b));
    return a.map((v, i) => mulArrays(v, b[i]));
}

function matmul(a: number[][], b: number[][]): number[][] {
    const m = a.length, k = a[0].length, n = b[0].length;
    const result: number[][] = Array.from({ length: m }, () => Array(n).fill(0));
    for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++)
            for (let x = 0; x < k; x++)
                result[i][j] += a[i][x] * b[x][j];
    return result;
}

function transpose(matrix: number[][]): number[][] {
    return matrix[0].map((_, col) => matrix.map(row => row[col]));
}

export class Tensor {
    public grad: Tensor | null = null;
    public operation: any = null;
    public parents: Tensor[] = [];

    public shape: number[];

    constructor(
        public data: any,
        public requires_grad = false,
        public device: 'cpu' | 'gpu' = 'cpu'
    ) {
        this.data = deepCopy(data);
        this.shape = shapeOf(data);
    }

    private isScalar(): boolean {
        return this.shape.length === 0 || (this.shape.length === 1 && this.shape[0] === 1);
    }

    static randn(shape: number[], requires_grad = false, device: 'cpu' | 'gpu' = 'cpu'): Tensor {
        const total = shape.reduce((a, b) => a * b, 1);
        const data: number[] = [];
        for (let i = 0; i < total; i += 2) {
            const u1 = Math.random(), u2 = Math.random();
            const r = Math.sqrt(-2 * Math.log(u1)), theta = 2 * Math.PI * u2;
            data.push(r * Math.cos(theta));
            if (i + 1 < total) data.push(r * Math.sin(theta));
        }
        return new Tensor(reshape(data, shape), requires_grad, device);
    }

    add(other: Tensor | number): Tensor {
        return new AddOp().forward(this, other);
    }

    mul(other: Tensor | number): Tensor {
        return new MulOp().forward(this, other);
    }

    matmul(other: Tensor): Tensor {
        return new MatMulOp().forward(this, other);
    }

    mean(): Tensor {
        return new MeanOp().forward(this);
    }

    backward(grad: Tensor | null = null): void {
        if (!this.requires_grad) return;
        if (!grad) {
            if (!this.isScalar()) throw new Error("Must provide grad for non-scalar tensors.");
            grad = new Tensor(1);
        }
        this.grad = this.grad
            ? new Tensor(addArrays(this.grad.data, grad.data))
            : new Tensor(deepCopy(grad.data));
        if (this.operation) this.operation.backward(grad, this);
    }
}

class AddOp {
    private forwardKernel = gpu.createKernel(function (a: number[][], b: number[][]) {
        return a[this.thread.y][this.thread.x] + b[this.thread.y][this.thread.x];
    })
        .setDynamicOutput(true)
        .setDynamicArguments(true);

    private backwardKernel = gpu.createKernel(function (grad: number[][]) {
        return grad[this.thread.y][this.thread.x];
    })
        .setDynamicOutput(true)
        .setDynamicArguments(true);

    forward(a: Tensor, b: Tensor | number): Tensor {
        const bData = typeof b === 'number' ? fillLike(a.data, b) : b.data;
        const [h, w] = a.shape;

        const result = a.device === 'gpu'
            ? toNestedArray(this.forwardKernel.setOutput([w, h])(a.data, bData)) as number[][]
            : addArrays(a.data, bData);

        const out = new Tensor(result, a.requires_grad || (b instanceof Tensor && b.requires_grad), a.device);
        out.operation = this;
        out.parents = [a, ...(b instanceof Tensor ? [b] : [])];
        return out;
    }

    backward(grad: Tensor, out: Tensor): void {
        const [a, b] = out.parents;
        const [h, w] = grad.shape;

        const gradData = grad.device === 'gpu'
            ? this.backwardKernel.setOutput([w, h])(grad.data) as number[][]
            : grad.data;

        if (a.requires_grad) a.backward(new Tensor(gradData, false, a.device));
        if (b instanceof Tensor && b.requires_grad) b.backward(new Tensor(gradData, false, b.device));
    }
}


class MulOp {
    forward(a: Tensor, b: Tensor | number): Tensor {
        const bData = typeof b === 'number' ? fillLike(a.data, b) : b.data;
        const out = new Tensor(mulArrays(a.data, bData), a.requires_grad || (b instanceof Tensor && b.requires_grad), a.device);
        out.operation = this;
        out.parents = [a, ...(b instanceof Tensor ? [b] : [])];
        return out;
    }

    backward(grad: Tensor, out: Tensor): void {
        const [a, b] = out.parents;
        if (a.requires_grad) {
            const bVal = b instanceof Tensor ? b.data : fillLike(a.data, b);
            a.backward(new Tensor(mulArrays(grad.data, bVal)));
        }
        if (b instanceof Tensor && b.requires_grad) {
            b.backward(new Tensor(mulArrays(grad.data, a.data)));
        }
    }
}

class MatMulOp {
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
            sum += grad[this.thread.y][i] * bT[this.thread.x][i];
        }
        return sum;
    }).setDynamicOutput(true).setDynamicArguments(true);

    private backwardBKernel = gpu.createKernel(function (aT: number[][], grad: number[][], m: number) {
        let sum = 0;
        for (let i = 0; i < m; i++) {
            sum += aT[this.thread.y][i] * grad[i][this.thread.x];
        }
        return sum;
    }).setDynamicOutput(true).setDynamicArguments(true);

    forward(a: Tensor, b: Tensor): Tensor {
        const [m, k1] = a.shape, [k2, n] = b.shape;
        if (k1 !== k2) throw new Error("Incompatible shapes for matmul");

        const result = a.device === 'gpu'
            ? toNestedArray(this.forwardKernel.setOutput([n, m])(a.data, b.data, k1))
            : matmul(a.data, b.data);

        const out = new Tensor(result, a.requires_grad || b.requires_grad, a.device);
        out.operation = this;
        out.parents = [a, b];
        return out;
    }

    backward(grad: Tensor, out: Tensor): void {
        const [a, b] = out.parents;

        if (a.requires_grad) {
            const bT = transpose(b.data);
            const dA = a.device === 'gpu'
                ? toNestedArray(this.backwardAKernel.setOutput([a.shape[1], a.shape[0]])(grad.data, bT, b.shape[1]))
                : matmul(grad.data, bT);
            a.backward(new Tensor(dA));
        }

        if (b.requires_grad) {
            const aT = transpose(a.data);
            const dB = b.device === 'gpu'
                ? toNestedArray(this.backwardBKernel.setOutput([b.shape[1], b.shape[0]])(aT, grad.data, a.shape[0]))
                : matmul(aT, grad.data);
            b.backward(new Tensor(dB));
        }
    }
}

class MeanOp {
    private count = 1;

    forward(input: Tensor): Tensor {
        const flat = flatDeep(input.data);
        this.count = flat.length;
        const out = new Tensor(flat.reduce((a, b) => a + b, 0) / this.count, input.requires_grad, input.device);
        out.operation = this;
        out.parents = [input];
        return out;
    }

    backward(grad: Tensor, out: Tensor): void {
        const [input] = out.parents;
        const g = Array.isArray(grad.data) ? flatDeep(grad.data)[0] : grad.data;
        input.backward(new Tensor(fillLike(input.data, g / this.count)));
    }
}
