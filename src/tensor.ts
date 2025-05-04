import { GPU } from 'gpu.js';

const gpu = new GPU();

function deepCopy<T>(x: T): T {
    return Array.isArray(x) ? (x as any[]).map(deepCopy) as any : x;
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
    // @ts-ignore
    return Array.isArray(arr) ? arr.flat(Infinity) : [arr];
}

function reshape(flat: number[], shape: number[]): any {
    if (shape.length === 0) return flat[0];
    const [dim, ...rest] = shape;
    const step = rest.reduce((a, b) => a * b, 1);
    const out: any[] = [];
    for (let i = 0; i < dim; i++) {
        out.push(reshape(flat.slice(i * step, (i + 1) * step), rest));
    }
    return out;
}

function addArrays(a: any, b: any): any {
    if (!Array.isArray(a) && !Array.isArray(b)) return a + b;
    if (!Array.isArray(a)) return addArrays(fillLike(b, a), b);
    if (!Array.isArray(b)) return addArrays(a, fillLike(a, b));
    return a.map((v: any, i: number) => addArrays(v, b[i]));
}

function mulArrays(a: any, b: any): any {
    if (!Array.isArray(a) && !Array.isArray(b)) return a * b;
    if (!Array.isArray(a)) return mulArrays(fillLike(b, a), b);
    if (!Array.isArray(b)) return mulArrays(a, fillLike(a, b));
    return a.map((v: any, i: number) => mulArrays(v, b[i]));
}

function fillLike(template: any, value: number): any {
    return Array.isArray(template)
        ? template.map(row => fillLike(row, value))
        : value;
}

function matmul(a: number[][], b: number[][]): number[][] {
    const m = a.length, k = a[0].length, n = b[0].length;
    // @ts-ignore
    const result: number[][] = Array.from({ length: m }, () => Array(n).fill(0));
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            for (let x = 0; x < k; x++) {
                result[i][j] += a[i][x] * b[x][j];
            }
        }
    }
    return result;
}

function transpose(matrix: number[][]): number[][] {
    return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
}

export class Tensor {
    public data: any;
    public shape: number[];
    public requires_grad: boolean;
    public grad: Tensor | null = null;
    public device: 'cpu' | 'gpu';
    public operation: any = null;
    public parents: Tensor[] = [];

    constructor(data: any, requires_grad = false, device: 'cpu' | 'gpu' = 'cpu') {
        this.data = deepCopy(data);
        this.shape = shapeOf(data);
        this.requires_grad = requires_grad;
        this.device = device;
    }

    private isScalar(): boolean {
        return this.shape.length === 0 || (this.shape.length === 1 && this.shape[0] === 1);
    }

    backward(grad: Tensor | null = null): void {
        if (!this.requires_grad) return;

        if (grad === null) {
            if (!this.isScalar()) throw new Error("Must provide grad for non-scalar tensors.");
            grad = new Tensor(1);
        }

        if (this.grad === null) {
            this.grad = new Tensor(deepCopy(grad.data));
        } else {
            this.grad.data = addArrays(this.grad.data, grad.data);
        }

        if (this.operation) {
            this.operation.backward(grad, this);
        }
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

    static randn(shape: number[], requires_grad = false, device: 'cpu' | 'gpu' = 'cpu'): Tensor {
        const total = shape.reduce((a, b) => a * b, 1);
        const data: number[] = [];

        for (let i = 0; i < total; i += 2) {
            const u1 = Math.random();
            const u2 = Math.random();
            const r = Math.sqrt(-2 * Math.log(u1));
            const theta = 2 * Math.PI * u2;
            data.push(r * Math.cos(theta));
            if (i + 1 < total) data.push(r * Math.sin(theta));
        }

        const reshaped = reshape(data, shape);
        return new Tensor(reshaped, requires_grad, device);
    }
}

class AddOp {
    private kernel: any;

    constructor() {
        this.kernel = gpu.createKernel(function (a: number[], b: number[]) {
            return a[this.thread.x] + b[this.thread.x];
        }).setOutput([1]); // Will override in .setOutput later
    }

    forward(a: Tensor, b: Tensor | number): Tensor {
        const useGPU = a.device === 'gpu';
        const bData = typeof b === 'number' ? fillLike(a.data, b) : (b as Tensor).data;
        let result;

        if (useGPU) {
            const flatA = flatDeep(a.data);
            const flatB = flatDeep(bData);
            const size = flatA.length;
            this.kernel.setOutput([size]);
            result = reshape(this.kernel(flatA, flatB) as number[], a.shape);
        } else {
            result = addArrays(a.data, bData);
        }

        const out = new Tensor(result, a.requires_grad || (b instanceof Tensor && b.requires_grad), a.device);
        out.operation = this;
        out.parents = [a, ...(b instanceof Tensor ? [b] : [])];
        return out;
    }

    backward(grad: Tensor, output: Tensor): void {
        const [a, b] = output.parents;
        if (a.requires_grad) a.backward(new Tensor(deepCopy(grad.data)));
        if (b?.requires_grad) b.backward(new Tensor(deepCopy(grad.data)));
    }
}

class MulOp {
    private kernel: any;

    constructor() {
        this.kernel = gpu.createKernel(function (a: number[], b: number[]) {
            return a[this.thread.x] * b[this.thread.x];
        }).setOutput([1]); // Will override in forward
    }

    forward(a: Tensor, b: Tensor | number): Tensor {
        const useGPU = a.device === 'gpu';
        const bData = typeof b === 'number' ? fillLike(a.data, b) : (b as Tensor).data;
        let result;

        if (useGPU) {
            const flatA = flatDeep(a.data);
            const flatB = flatDeep(bData);
            const size = flatA.length;
            this.kernel.setOutput([size]);
            result = reshape(this.kernel(flatA, flatB) as number[], a.shape);
        } else {
            result = mulArrays(a.data, bData);
        }

        const out = new Tensor(result, a.requires_grad || (b instanceof Tensor && b.requires_grad), a.device);
        out.operation = this;
        out.parents = [a, ...(b instanceof Tensor ? [b] : [])];
        return out;
    }

    backward(grad: Tensor, output: Tensor): void {
        const [a, b] = output.parents;
        if (a.requires_grad) {
            const bData = b instanceof Tensor ? b.data : fillLike(a.data, b);
            a.backward(new Tensor(mulArrays(grad.data, bData)));
        }
        if (b instanceof Tensor && b.requires_grad) {
            b.backward(new Tensor(mulArrays(grad.data, a.data)));
        }
    }
}


class MatMulOp {
    private kernel: any;

    constructor() {
        this.kernel = gpu.createKernel(function (a: number[], b: number[], k: number, n: number) {
            let sum = 0;
            const row = Math.floor(this.thread.x / n);
            const col = this.thread.x % n;
            for (let i = 0; i < k; i++) {
                sum += a[row * k + i] * b[i * n + col];
            }
            return sum;
        }).setOutput([1]); // Set actual output in forward
    }

    forward(a: Tensor, b: Tensor): Tensor {
        const [m, k1] = a.shape;
        const [k2, n] = b.shape;
        if (k1 !== k2) throw new Error("Incompatible shapes for matmul");

        let result;
        if (a.device === 'gpu') {
            const flatA = flatDeep(a.data);
            const flatB = flatDeep(b.data);
            this.kernel.setOutput([m * n]);
            result = reshape(this.kernel(flatA, flatB, k1, n) as number[], [m, n]);
        } else {
            result = matmul(a.data, b.data);
        }

        const out = new Tensor(result, a.requires_grad || b.requires_grad, a.device);
        out.operation = this;
        out.parents = [a, b];
        return out;
    }

    backward(grad: Tensor, output: Tensor): void {
        const [a, b] = output.parents;
        if (a.requires_grad) a.backward(new Tensor(matmul(grad.data, transpose(b.data))));
        if (b.requires_grad) b.backward(new Tensor(matmul(transpose(a.data), grad.data)));
    }
}

class MeanOp {
    private count: number = 1;
    private kernel: any;

    constructor() {
        this.kernel = gpu.createKernel(function (x: number[]) {
            return x[this.thread.x];
        }).setOutput([1]);
    }

    forward(input: Tensor): Tensor {
        const flat = flatDeep(input.data);
        this.count = flat.length;

        let sum: number;
        if (input.device === 'gpu') {
            this.kernel.setOutput([flat.length]);
            const out = this.kernel(flat) as number[];
            sum = out.reduce((a, b) => a + b, 0);
        } else {
            sum = flat.reduce((a, b) => a + b, 0);
        }

        const meanVal = sum / this.count;
        const out = new Tensor(meanVal, input.requires_grad, input.device);
        out.operation = this;
        out.parents = [input];
        return out;
    }

    backward(grad: Tensor, output: Tensor): void {
        const [input] = output.parents;
        const gradVal = Array.isArray(grad.data) ? flatDeep(grad.data)[0] : grad.data;
        const scaledGrad = gradVal / this.count;
        input.backward(new Tensor(fillLike(input.data, scaledGrad)));
    }
}
