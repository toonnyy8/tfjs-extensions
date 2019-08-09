import * as tf from "@tensorflow/tfjs"

export function repeatElements(x, rep, axis) {
    let reps = [...Array(x.shape.length)].map(_ => 1)
    reps[axis] = rep
    return tf.tile(x, reps)
}

export function batchDot(x, y, axis) {
    return tf.sum(tf.mul(x, y), axis)
}

export function mergeShape(tensor = tf.tensor(), axises, at = null) {
    return tf.tidy(() => {
        let shape = tensor.shape
        let merge = -1
        let transposeShape = new Array(shape.length).fill(0).map((val, idx) => idx)
        let newShape = null
        if (axises && axises.length != 0) {
            newShape = []
            at = at || axises[0]
            if (axises.find((axis) => axis == at) != undefined) {
                transposeShape.splice(at, 1, axises.slice())
            } else {
                console.error("axis ${at} is not at axises")
            }

            axises.sort(function (a, b) {//由大到小排序
                if (a > b) return -1;
                if (a < b) return 1;
                return 0;
            });
            axises.forEach((axis) => {
                if (!Array.isArray(transposeShape[axis])) {
                    transposeShape.splice(axis, 1)
                }
            })
            transposeShape = transposeShape.flat()
            shape.forEach((val, idx) => {
                if (axises.find(axis => axis == idx) != undefined) {
                    if (idx == at) {
                        merge = 1
                        axises.forEach(axis => {
                            merge *= shape[axis]
                        })
                        newShape.push(merge)
                    }
                } else {
                    newShape.push(val)
                }
            });
        }
        // console.log(transposeShape)
        // tensor.print()
        // tensor.transpose(transposeShape).print()
        return tensor.transpose(transposeShape).reshape(newShape || shape)
    })
}

export function einsum(subscripts = "", ...operands) {
    let [equation, inputs, , output] = subscripts.match("^([a-zA-Z,.]+)(->)([a-zA-Z.]*)?$") || [null, null, null, null]

    if (operands.find(input => !(input instanceof tf.Tensor)) != undefined || operands.length == 0) {
        console.error(`operands type is not tensor`)
        return
    }

    if (!equation) {
        console.error(`Indices have incorrect format: ${subscripts}`)
        return
    }

    inputs = inputs.split(",")
    if (inputs.find(val => val == "") != undefined) {
        console.error(`Indices have incorrect format: ${subscripts}`)
        return
    }

    if (inputs.length != operands.length) {
        console.error(`Incorrect number of operands`)
        return
    }

    inputs.forEach((_, idx, arr) => {
        arr[idx] = arr[idx].split("")
        // console.log(Math.max(...arr[idx]))
        arr[idx].forEach(() => {

        })
    })
    console.log(equation)

    return tf.tidy(() => {
        let a = mergeShape(operands[0], [1, 2])
        let b = mergeShape(operands[1], [1, 2])

        return tf.sum(
            mergeShape(
                tf.transpose(
                    tf.squeeze(
                        tf.stack(
                            tf.unstack(
                                tf.expandDims(
                                    mergeShape(
                                        tf.transpose(a, [0, 1, 2])
                                        , [0, 2], 2
                                    )
                                    , [0]
                                )
                            ).map(
                                (t) => {
                                    return tf.expandDims(
                                        t.mul(
                                            mergeShape(
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
    })
}