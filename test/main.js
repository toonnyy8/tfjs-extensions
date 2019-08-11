import "@babel/polyfill"

import * as tf from "@tensorflow/tfjs"
import * as tfex from "../src"
import { async } from "q";
import { TFHUB_SEARCH_PARAM } from "@tensorflow/tfjs-converter/dist/src/executor/graph_model";
import { reshape } from "@tensorflow/tfjs-layers/dist/exports_layers";

(async () => {
    let a = tf.tensor([
        [
            [1, 4, 2],
            [1, 2, 1]
        ],
        [
            [1, 4, 3],
            [1, 3, 1]
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

    console.log("---------")
    console.log("tf.einsum('ijk,gkh->k',a,b) : [24, 78, 84]")
    await testTime(() => {
        tfex.einsum('ijk,gkh->k', a, b).print()
    }, "tf.einsum('ijk,gkh->k',a,b)")
    console.log("---------")
    console.log("tf.einsum('ijk,imj->i',a,b) : [70 67]")
    await testTime(() => {
        tfex.einsum('ijk,imj->i', a, b).print()
    }, "tf.einsum('ijk,imj->i',a,b)")
    console.log("---------")
    console.log("tf.einsum('ijk,iml->i',a,b) : [143 143]")
    await testTime(() => {
        tfex.einsum('ijk,iml->', a, b).print()
    }, "tf.einsum('ijk,iml->',a,b)")
    console.log("---------")
})()

async function testTime(f = () => { }, msg = "msg") {
    const time = await tf.time(f)
    console.log(`${msg}--kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`);
}