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

    console.log(tf)
    console.log("---------")

    console.log("---------")
    // tf.mul(tfex.mergeShape(a, [1, 2]), tfex.mergeShape(b, [1, 2])).sum(1).print()//tf.einsum('ijk,ijk->i', a, b)

    // await testTime(() => {
    //     tf.print(
    //         tf.sum(
    //             tfex.mergeShape(
    //                 tf.transpose(
    //                     tf.squeeze(
    //                         tf.stack(
    //                             tf.unstack(
    //                                 tf.expandDims(
    //                                     tfex.mergeShape(a, [1, 2])
    //                                     , [0]
    //                                 )
    //                             ).map(
    //                                 (t1) => {
    //                                     return t1.mul(tfex.mergeShape(b, [1, 2]))
    //                                 }
    //                             )
    //                         )
    //                         , [0]
    //                     )
    //                     , [0, 1]
    //                 )
    //                 , []
    //             )
    //             , [1]
    //         )
    //     )
    // }, "tf.einsum('ijk,ijk->i', a, b)")//[27, 20]


    await testTime(() => {
        tf.print(
            tf.sum(
                tfex.mergeShape(
                    tf.transpose(
                        tf.squeeze(
                            tf.stack(
                                tf.unstack(
                                    tf.expandDims(
                                        tfex.mergeShape(a, [1, 2])
                                        , [0]
                                    )
                                ).map(
                                    (t1) => {
                                        return tf.expandDims(t1.mul(tfex.mergeShape(b, [1, 2])), [2])
                                    }
                                )
                            )
                            , [0, 3]
                        )
                        , [0, 1]
                    )
                    , []
                )
                , [1]
            )
        )
    }, "tf.einsum('ijk,ijk->i', a, b)")//[27, 20]

    console.log("---------")
    // await testTime(() => {
    //     tfex.mergeShape(tf.stack(tfex.mergeShape(a, [1, 2]).unstack().map((t1, idx1) => {
    //         return t1.mul(tfex.mergeShape(b, [1, 2]))
    //     })).transpose([1, 0, 2]), [1, 2]).sum(1).print()
    // }, "tf.einsum('hjk,ijk->i', a, b)")//[59, 37]

    // await testTime(() => {
    //     tf.print(
    //         tf.sum(
    //             tfex.mergeShape(
    //                 tf.transpose(
    //                     // tf.squeeze(
    //                     tf.stack(
    //                         tf.unstack(
    //                             // tf.expandDims(
    //                             tfex.mergeShape(a, [1, 2])
    //                             //     , [0]
    //                             // )
    //                         ).map(
    //                             (t1) => {
    //                                 return t1.mul(tfex.mergeShape(b, [1, 2]))
    //                             }
    //                         )
    //                     )
    //                     //     , [0]
    //                     // )
    //                     , [1, 0, 2]
    //                 )
    //                 , [1, 2]
    //             )
    //             , [1]
    //         )
    //     )
    // }, "tf.einsum('hjk,ijk->i', a, b)")//[59, 37]

    await testTime(() => {
        tf.print(
            tf.sum(
                tfex.mergeShape(
                    tf.transpose(
                        tf.squeeze(
                            tf.stack(
                                tf.unstack(
                                    tf.expandDims(
                                        tfex.mergeShape(a, [1, 2])
                                        , [1]
                                    )
                                ).map(
                                    (t1) => {
                                        return tf.expandDims(t1.mul(tfex.mergeShape(b, [1, 2])), [2])
                                    }
                                )
                            )
                            , [3]
                        )
                        , [1, 0, 2]
                    )
                    , [1, 2]
                )
                , [1]
            )
        )
    }, "tf.einsum('hjk,ijk->i', a, b)")//[59, 37]
    // tfex.einsum('hjk,ijk->i', a, b).print()
    console.log("---------")
    // await testTime(() => {
    //     tfex.mergeShape(tf.stack(tfex.mergeShape(a, [1, 2]).unstack().map((t1, idx1) => {
    //         return t1.mul(tfex.mergeShape(b, [1, 2]))
    //     })).transpose([0, 1, 2]), [1, 2]).sum(1).print()
    // }, "tf.einsum('hjk,ijk->h', a, b)")//[44, 52]

    await testTime(() => {
        tf.print(
            tf.sum(
                tfex.mergeShape(
                    tf.transpose(
                        tf.squeeze(
                            tf.stack(
                                tf.unstack(
                                    tf.expandDims(
                                        tfex.mergeShape(a, [1, 2])
                                        , [1]
                                    )
                                ).map(
                                    (t1) => {
                                        return tf.expandDims(t1.mul(tfex.mergeShape(b, [1, 2])), [2])
                                    }
                                )
                            )
                            , [3]
                        )
                        , [0, 1, 2]
                    )
                    , [1, 2]
                )
                , [1]
            )
        )
    }, "tf.einsum('hjk,ijk->h', a, b)")//[44, 52]
    console.log("---------")
    // tf.matMul(tfex.mergeShape(a, [1, 2]), tfex.mergeShape(b, [1, 2]).transpose()).sum(1).print()//tf.einsum('ijk,hjk->i', a, b)
    await testTime(() => {
        tf.print(
            tf.sum(
                tfex.mergeShape(
                    tf.transpose(
                        tf.squeeze(
                            tf.stack(
                                tf.unstack(
                                    tf.expandDims(
                                        tfex.mergeShape(tf.transpose(a, [0, 1, 2]), [1, 2])
                                        , [0]
                                    )
                                ).map(
                                    (t1) => {
                                        return tf.expandDims(t1.mul(tfex.mergeShape(tf.transpose(b, [0, 2, 1]), [1, 2])), [2])
                                    }
                                )
                            )
                            , [0, 3]
                        )
                        , [0, 1]
                    )
                    , []
                )
                , [1]
            )
        )
    }, "tf.einsum('ijk,ikj->i',a,b)")//[23 22]
    console.log("---------")

    await testTime(() => {
        tf.print(
            tf.sum(
                tfex.mergeShape(
                    tf.transpose(
                        tf.squeeze(
                            tf.stack(
                                tf.unstack(
                                    tf.expandDims(
                                        tfex.mergeShape(
                                            tf.transpose(a, [0, 1, 2])
                                            , [0, 2], 2
                                        )
                                        , [0]
                                    )
                                ).map(
                                    (t1) => {
                                        return tf.expandDims(
                                            t1.mul(
                                                tfex.mergeShape(
                                                    tf.transpose(b, [0, 2, 1])
                                                    , [0, 2], 2
                                                )
                                            ), [2])
                                    }
                                )
                            )
                            , [0, 3]
                        )
                        , [0, 1]
                    )
                    , []
                )
                , [1]
            )
        )
    }, "tf.einsum('ijk,ikj->j',a,b)")//[26 19]
    console.log("---------")
    await testTime(() => {
        a.reshape([-1, 1]).dot(b.reshape([1, -1])).reshape([2, 2, 3, 2, 3, 2])

    }, "tf.einsum('ijk,lmn->ijklmn',a,b)")//[26 19]

    console.log("---------")
    // console.log("---------")
    // a.reshape([-1]).mul(b.reshape([-1])).reshape([2, -1]).sum(1).print()//tf.einsum('ijk,ijk->i', a, b)
    // console.log("---------")
    // a.reshape([2, -1]).dot(b.reshape([2, -1]).transpose()).sum(0).print()//tf.einsum('hjk,ijk->i', a, b)
    // console.log("---------")
    // a.reshape([2, -1]).mul(b.transpose([0, 2, 1]).reshape([2, -1])).sum(1).print()//tf.einsum('ijk,ikj->i',a,b)
    // console.log("---------")
    // a.transpose([1, 0, 2]).reshape([2, -1]).mul(b.transpose([0, 2, 1]).transpose([1, 0, 2]).reshape([2, -1])).sum(1).print()//tf.einsum('ijk,ikj->j',a,b)
    // console.log("---------")
    // a.reshape([-1, 1]).dot(b.reshape([1, -1])).reshape([2, 2, 3, 2, 3, 2]).print()
    console.log("---------")
    console.log("tf.einsum('ijk,gkh->k',a,b) : [24, 78, 84]")
    a.reshape([-1, 3]).unstack(0).forEach((t) => t.reshape([-1]).mul(b.transpose([0, 2, 1]).reshape([-1, 3])).sum(0).print())
    console.log("---------")

    console.log("tf.einsum('ijk,imj->i',a,b) : [70 67]")
    let b_ = b.transpose([0, 2, 1])
    await testTime(() => {
        a.reshape([-1, 1]).dot(b_.reshape([1, -1])).reshape(a.shape.concat(b_.shape)).sum([2, 5]).reshape([-1]).gather(tf.range(0, 4 * 4, 4 + 1, "int32")).reshape([2, 2]).sum(1).print()

    }, "tf.einsum('ijk,imj->i',a,b)")
    console.log("---------")
    console.log("tf.einsum('ijk,iml->i',a,b) : [143 143]")
    await testTime(() => {
        // a.reshape([-1, 1]).dot(b_.reshape([1, -1])).reshape(a.shape.concat(b_.shape)).sum([1, 2, 4, 5]).reshape([-1]).gather(tf.range(0, 2 * 2, 2 + 1, "int32")).reshape([-1]).sum([]).print()
        // a.reshape([-1, 1]).dot(b_.reshape([1, -1])).reshape(a.shape.concat(b_.shape)).sum([1, 2, 4, 5]).print()
        tfex.einsum('ikj,iml->i', a, b).print()
    }, "tf.einsum('ijk,iml->i',a,b)")

    console.log("---------")
})()

async function testTime(f = () => { }, msg = "msg") {
    const time = await tf.time(f)
    console.log(`${msg}--kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`);
}