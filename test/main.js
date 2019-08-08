import "@babel/polyfill"

import * as tf from "@tensorflow/tfjs"
import * as tfex from "../src"

(async () => {
    let a = tf.tensor([
        [
            [1, 2, 3],
            [1, 2, 1]
        ],
        [
            [1, 2, 3],
            [1, 2, 1]
        ]
    ])
    let b = tf.tensor([
        [
            [1, 3],
            [2, 1],
            [3, 3]
        ],
        [
            [1, 1],
            [1, 2],
            [2, 4]
        ]
    ])

    console.log(tf)
    console.log("---------")
    a.print()
    console.log("---------")

    a.sum(2).sum(1).print()
    const time1 = await tf.time(() => a.sum(2).sum(1));
    console.log(`kernelMs: ${time1.kernelMs}, wallTimeMs: ${time1.wallMs}`);

    console.log("---------")

    a.reshape([2, 6]).sum(1).print()
    const time2 = await tf.time(() => a.reshape([2, 6]).sum(1));
    console.log(`kernelMs: ${time2.kernelMs}, wallTimeMs: ${time2.wallMs}`);

    console.log("---------")

    console.log(a.shape)
    tfex.mergeShape(a, [1, 2]).sum(1).print()
    const time3 = await tf.time(() => tfex.mergeShape(a, [1, 2]).sum(1));
    console.log(`kernelMs: ${time3.kernelMs}, wallTimeMs: ${time3.wallMs}`);
    tfex.mergeShape(b, [1, 2]).transpose().print()
    console.log("---------")
    tf.mul(tfex.mergeShape(a, [1, 2]), tfex.mergeShape(b, [1, 2])).sum(1).print()//tf.einsum('ijk,ijk->i', a, b)
    console.log("---------")
    tf.dot(tfex.mergeShape(a, [1, 2]), tfex.mergeShape(b, [1, 2]).transpose()).sum(0).print()//tf.einsum('hjk,ijk->i', a, b)
    console.log("---------")
    tf.matMul(tfex.mergeShape(a, [1, 2]), tfex.mergeShape(b, [1, 2]).transpose()).sum(1).print()//tf.einsum('ijk,hjk->i', a, b)
    console.log("---------")
    tf.mul(tfex.mergeShape(a, [0, 2]), tfex.mergeShape(b.transpose([0, 2, 1]), [0, 2])).sum(0).print()//tf.einsum('ijk,hjk->i', a, b)

})()