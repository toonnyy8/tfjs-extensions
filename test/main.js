import "@babel/polyfill"

import * as tf from "@tensorflow/tfjs"
import * as tfex from "../src"

// (async() => {
//     console.log(tf.memory())
//     let a = tf.tensor([
//         [
//             [1, 4, 2],
//             [1, 2, 1]
//         ],
//         [
//             [1, 4, 3],
//             [1, 3, 1]
//         ]
//     ])
//     let b = tf.tensor([
//         [
//             [1, 3],
//             [2, 1],
//             [3, 3]
//         ],
//         [
//             [1, 1],
//             [1, 2],
//             [2, 4]
//         ]
//     ])

//     console.log("---------")
//     console.log("tf.einsum('ijk,gkh->k',a,b) : [24, 78, 84]")
//     await testTime(() => {
//         tfex.einsum('ijk,gkh->k', a, b).print()
//     }, "tf.einsum('ijk,gkh->k',a,b)")
//     console.log("---------")
//     console.log("tf.einsum('ijk,imj->i',a,b) : [70 67]")
//     await testTime(() => {
//         tfex.einsum('ijk,imj->i', a, b).print()
//     }, "tf.einsum('ijk,imj->i',a,b)")
//     console.log("---------")
//     console.log("tf.einsum('ijk,iml->i',a,b) : [143 143]")
//     await testTime(() => {
//         tfex.einsum('ijk,iml->', a, b).print()
//     }, "tf.einsum('ijk,iml->',a,b)")
//     console.log("---------")
//     await testTime(() => {
//         tfex.einsum('ijk,imj->ij', a, b).print()
//     }, "tf.einsum('ijk,imj->ij',a,b)")
//     console.log("---------")
//     console.log(tf.memory())

// })()

// async function testTime(f = () => {}, msg = "msg") {
//     const time = await tf.time(f)
//     console.log(`${msg}--kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`);
// }

// let a = tfex.layers.lambda({ func: (x, y) => { return [tf.add(x, y), tf.add(x, y), tf.add(x, y)] } })
// console.log(tf.memory())
// console.log(
//     a.apply([tf.input({ shape: [3] }), tf.input({ shape: [3] })])
// )
// console.log(tf.memory())

console.log(tf.memory())

let vS = tfex.scope.variableScope("G")

vS.with((scope) => {
    scope.variableScope("FF").with((scope, n) => {
        let a = scope.getVariable("qkv", [1, 1, 3])
        a.print()
        tfex.scope.variableScope(n + "layers").with((_, n) => {
            let a = _.getVariable("qkv", [1, 3, 1])
            _.getVariable("qkv2", [1, 3, 1])
        })
    })
    let a = scope.getVariable("qkv", [3, 1, 1])
})
let c
vS.with((scope) => {
    return scope.variableScope("FF").with((scope, n) => {
        return tfex.scope.variableScope(n + "layers").with((_, n) => {
            _.dispose("qkv2")
            c = tf.tensor([1])
            return c
        })
    })
})
c.print()

console.log(tf.memory())
