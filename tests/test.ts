import { GPU } from 'gpu.js';
const gpu = new GPU();
function toNestedArray(data: any): number[][] {
    return Array.isArray(data) ? data.map((row: any) => Array.from(row)) : [];
}
const testKernel = gpu.createKernel(function(a, b) {
    return a[this.thread.y][this.thread.x] + b[this.thread.y][this.thread.x];
}).setOutput([3, 2]);

let a = testKernel([
    [1, 2, 3],
    [4, 5, 6]
], [
    [10, 20, 30],
    [40, 50, 60]
]);

console.log(toNestedArray(a))