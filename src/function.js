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
    let _
    [, subscript.inputs, _, subscript.output] = subscripts.match("^([a-zA-Z,.]+)(->)?([a-zA-Z.]*)?$") || [null, null, null, null]

    if (_ == null) {
        console.error(`Need "->"`)
        return
    }

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

    if (subscript.output == null) {
        subscript.output = ""
    }

    if (subscript.inputs.length == 1) {
        return einsumSingleInput(subscript, operands[0])
    } else {
        return einsumMultipleInput(subscript, operands)
    }
}

function einsumSingleInput(subscript = { inputs: [""], output: "" }, operand = tf.tensor()) {
    return tf.tidy(() => {
        let inputInfo = subscript.inputs[0]
            .split("")
            .map((tag, axis) => {
                return { tag: tag, axis: axis, dim: operand.shape[axis] }
            })
            .sort((a, b) => {//由小到大排序
                if (a.tag > b.tag) return 1;
                if (a.tag < b.tag) return -1;
                return 0;
            })

        let outputInfo = subscript.output == "" ? [] : subscript.output
            .split("")
            .map((tag, axis) => {
                if (subscript.inputs[0].search(tag) == -1) {
                    console.error(`Output Tag_${tag} does not exist`)
                }
                return { tag: tag, axis: axis }
            })
            .sort((a, b) => {//由小到大排序
                if (a.tag > b.tag) return 1;
                if (a.tag < b.tag) return -1;
                return 0;
            })

        let diadInfo = inputInfo.reduce((last, info) => {
            if (last[info.tag]) {
                if (last[info.tag][0] != info.dim) {
                    console.error(`Dim of Tag_${info.tag} are inconsistent`)
                } else {
                    last[info.tag].push(info.dim)
                }
            } else {
                last[info.tag] = [info.dim]
            }
            return last
        }, {})

        let diagShape = Object.values(diadInfo)

        let [indices, newShape] = diagIndices(diagShape)
        newShape = newShape.flat()

        let tagSum = Object.keys(diadInfo)
            .reduce((last, tag, axis) => {
                if (subscript.output.search(tag) == -1) {
                    last.push(axis)
                }
                return last
            }, [])

        return operand.
            transpose(inputInfo.map((info) => info.axis))
            .reshape([-1])
            .gather(indices)
            .reshape(newShape)
            .sum(tagSum)
            .transpose(outputInfo.map((info) => info.axis))
    })
}
function diagIndices(diag = [[]]) {
    let diag_ = JSON.parse(JSON.stringify(diag))
    return tf.tidy(() => {
        let indices = tf.ones([1], "int32")
        diag_.forEach((tagsDim) => {
            let dim = tagsDim[0]
            if (tagsDim.length > 1) {
                for (; tagsDim.length > 1; dim = tagsDim.pop()) {
                    indices = indices
                        .reshape([-1, 1])
                        .dot(tf.tidy(() => {
                            let indices = tf.range(0, dim ** 2, dim + 1, "int32")
                            return tf.scatterND(indices, tf.ones(indices.shape, "int32"), [dim ** 2])
                        })
                            .reshape([1, -1])
                        )
                }
            } else {
                indices = indices
                    .reshape([-1, 1])
                    .dot(tf.ones([1, dim], "int32"))
            }
        })
        let indicesBuffer = indices.reshape([-1]).bufferSync()
        indices.dispose()
        indices = []
        for (let i = 0; i < indicesBuffer.size; i++) {
            if (indicesBuffer.values[i]) {
                indices.push(i)
            }
        }
        indices = tf.tensor(indices).cast("int32")
        return [indices, diag_]
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

        let [x, y] = [
            operands.shift().transpose(inputInfo.x.map((info) => info.axis)),
            operands.shift().transpose(inputInfo.y.map((info) => info.axis))
        ]

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
        subscript.inputs.unshift(
            inputInfo.x
                .reduce((last, info) => last + info.tag, "")
                .concat(
                    inputInfo.y
                        .reduce((last, info) => last + info.tag, "")
                )
        )

        if (subscript.inputs.length == 1) {
            return einsumSingleInput(subscript, operands[0])
        } else {
            return einsumMultipleInput(subscript, operands)
        }
    })
}