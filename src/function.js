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
    let subscript = {
        inputs: null,
        output: null
    };

    [, subscript.inputs, , subscript.output] = subscripts.match("^([a-zA-Z,.]+)(->)?([a-zA-Z.]*)?$") || [null, null, null, null]

    if (!subscript.inputs) {
        console.error(`Indices have incorrect format: ${subscripts}`)
        return
    }

    if (operands.find(input => !(input instanceof tf.Tensor)) != undefined || operands.length == 0) {
        console.error(`operands type is not tensor`)
        return
    }

    subscript.inputs = subscript.inputs.split(",")
    if (subscript.inputs.find(val => val == "") != undefined) {
        console.error(`Indices have incorrect format: ${subscripts}`)
        return
    }

    if (subscript.inputs.length != operands.length) {
        console.error(`Incorrect number of operands`)
        return
    }

    if (subscript.inputs.length == 1) {
        return einsumSingleInput(subscript, operands[0])
    } else {
        return einsumMultipleInput(subscript, operands)
    }

    subscript.inputs[0] = subscript.inputs[0].split("").map((val, idx) => {
        return { target: val, axis: idx, common: false }
    })
    subscript.inputs[0].sort(function (a, b) {//由小到大排序
        if (a.target > b.target) return 1;
        if (a.target < b.target) return -1;
        return 0;
    });

    if (subscript.inputs.length >= 2) {
        subscript.inputs[1] = subscript.inputs[1].split("").map((val, idx) => {
            return { target: val, axis: idx, common: false }
        })
        subscript.inputs[1].sort(function (a, b) {//由小到大排序
            if (a.target > b.target) return 1;
            if (a.target < b.target) return -1;
            return 0;
        });
        subscript.inputs[0].forEach((info) => {
            info.common = subscript.inputs[1].find((info_) => {
                return info.target == info_.target
            }) != undefined
        })
        subscript.inputs[1].forEach((info) => {
            info.common = subscript.inputs[0].find((info_) => {
                return info.target == info_.target
            }) != undefined
        })
    }

    console.log(subscript.inputs)

    subscript.output = subscript.output.split("").map((val, idx) => {
        return { target: val, axis: idx }
    })
    subscript.output.sort(function (a, b) {//由小到大排序
        if (a.target > b.target) return 1;
        if (a.target < b.target) return -1;
        return 0;
    });
    console.log(subscript.output)

    return tf.tidy(() => {
        let i = []
        operands.forEach((t, idx) => [
            i.push(t.transpose(subscript.inputs[idx].map((info) => info.axis)))
        ])
        let sum = subscript.inputs.length >= 2 ?
            subscript.inputs[0]
                .map((info, axis) => !info.common ? axis : undefined)
                .concat(subscript.inputs[1].map((info, axis) => !info.common ? axis + subscript.inputs[0].length : undefined))
                .filter(function (el) {
                    return el != null;
                }) : []
        let step = subscript.inputs.length >= 2 ? subscript.inputs[0].reduce((last, info, axis) => {
            return info.common ? last * i[0].shape[axis] : last
        }, 1) + 1 : 1
        let stop = subscript.inputs.length >= 2 ? (step - 1) ** 2 : subscript.inputs[0].reduce((last, info, axis) => {
            return last * i[0].shape[axis]
        }, 1)

        console.log(step)

        // if (subscript.inputs.length == 1) {
        //     return i[0]
        //         .sum(sum)
        //         .reshape([-1])
        //         .gather(tf.range(0, stop, step, "int32"))
        //         .reshape([-1])
        //         .sum([])
        //     // .transpose([])
        // } else {
        //     return einsum("",
        //         i[0]
        //             .reshape([-1, 1])
        //             .dot(i[1].reshape([1, -1]))
        //             .reshape(i[0].shape.concat(i[1].shape)),
        //         ...operands
        //     )
        // }

        if (subscript.inputs.length == 1) {
            return i[0]
                .sum(sum)
            // .transpose([])
        } else if (subscript.inputs.length == 2) {
            return i[0]
                .reshape([-1, 1])
                .dot(i[1].reshape([1, -1]))
                .reshape(i[0].shape.concat(i[1].shape))
                .sum(sum)
                .reshape([-1])
                .gather(tf.range(0, stop, step, "int32"))
                .reshape([-1])
                .sum([])
            // .transpose([])
        } else {
            return einsum("",
                i[0]
                    .reshape([-1, 1])
                    .dot(i[1].reshape([1, -1]))
                    .reshape(i[0].shape.concat(i[1].shape)),
                ...operands
            )
        }
    })
}

function einsumSingleInput(subscript = { inputs: null, output: null }, operand = tf.tensor()) {
    return tf.tidy(() => {
        return operand
            .sum(sum)
            .reshape([-1])
            .gather(tf.range(0, stop, step, "int32"))
            .reshape([-1])
            .sum([])
            .transpose([])
    })
}

function einsumMultipleInput(subscript = { inputs: [""], output: null }, operands = [tf.tensor()]) {
    return tf.tidy(() => {
        let inputInfo = {
            x: subscript.inputs
                .shift()
                .split("")
                .map((tag, axis) => {
                    return { tag: tag, axis: axis }
                })
                .sort((a, b) => {//由小到大排序
                    if (a.tag > b.tag) return 1;
                    if (a.tag < b.tag) return -1;
                    return 0;
                }),
            y: subscript.inputs
                .shift()
                .split("")
                .map((tag, axis) => {
                    return { tag: tag, axis: axis }
                })
                .sort((a, b) => {//由小到大排序
                    if (a.tag > b.tag) return 1;
                    if (a.tag < b.tag) return -1;
                    return 0;
                })
        }

        let [x, y] = [operands.shift().transpose(), operands.shift().transpose()]
        operands.unshift(
            x
                .reshape([-1, 1])
                .dot(y.reshape([1, -1]))
                .reshape(x.shape.concat(y.shape))
            // .sum(sum)
            // .reshape([-1])
            // .gather(tf.range(0, stop, step, "int32"))
            // .reshape([-1])
            // .sum([])
            // .transpose([])
        )
        if (subscript.inputs.length == 1) {
            return einsumSingleInput(subscript, operands[0])
        } else {
            return einsumMultipleInput(subscript, operands)
        }
    })
}