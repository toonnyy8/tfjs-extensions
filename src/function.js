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
    let [equation, inputsInfo, , outputInfo] = subscripts.match("^([a-zA-Z,.]+)(->)([a-zA-Z.]*)?$") || [null, null, null, null]

    if (operands.find(input => !(input instanceof tf.Tensor)) != undefined || operands.length == 0) {
        console.error(`operands type is not tensor`)
        return
    }

    if (!equation) {
        console.error(`Indices have incorrect format: ${subscripts}`)
        return
    }

    inputsInfo = inputsInfo.split(",")
    if (inputsInfo.find(val => val == "") != undefined) {
        console.error(`Indices have incorrect format: ${subscripts}`)
        return
    }

    if (inputsInfo.length != operands.length) {
        console.error(`Incorrect number of operands`)
        return
    }

    inputsInfo[0] = inputsInfo[0].split("").map((val, idx) => {
        return { target: val, axis: idx, common: false }
    })
    inputsInfo[0].sort(function (a, b) {//由小到大排序
        if (a.target > b.target) return 1;
        if (a.target < b.target) return -1;
        return 0;
    });

    if (inputsInfo.length >= 2) {
        inputsInfo[1] = inputsInfo[1].split("").map((val, idx) => {
            return { target: val, axis: idx, common: false }
        })
        inputsInfo[1].sort(function (a, b) {//由小到大排序
            if (a.target > b.target) return 1;
            if (a.target < b.target) return -1;
            return 0;
        });
        inputsInfo[0].forEach((info) => {
            info.common = inputsInfo[1].find((info_) => {
                return info.target == info_.target
            }) != undefined
        })
        inputsInfo[1].forEach((info) => {
            info.common = inputsInfo[0].find((info_) => {
                return info.target == info_.target
            }) != undefined
        })
    }

    console.log(inputsInfo)

    outputInfo = outputInfo.split("").map((val, idx) => {
        return { target: val, axis: idx }
    })
    outputInfo.sort(function (a, b) {//由小到大排序
        if (a.target > b.target) return 1;
        if (a.target < b.target) return -1;
        return 0;
    });
    console.log(outputInfo)

    return tf.tidy(() => {
        let i = []
        operands.forEach((t, idx) => [
            i.push(t.transpose(inputsInfo[idx].map((info) => info.axis)))
        ])
        let sum = inputsInfo.length >= 2 ?
            inputsInfo[0]
                .map((info, axis) => !info.common ? axis : undefined)
                .concat(inputsInfo[1].map((info, axis) => !info.common ? axis + inputsInfo[0].length : undefined))
                .filter(function (el) {
                    return el != null;
                }) : []
        let step = inputsInfo.length >= 2 ? inputsInfo[0].reduce((last, info, axis) => {
            return info.common ? last * i[0].shape[axis] : last
        }, 1) + 1 : 1
        let stop = inputsInfo.length >= 2 ? (step - 1) ** 2 : inputsInfo[0].reduce((last, info, axis) => {
            return last * i[0].shape[axis]
        }, 1)

        console.log(step)

        return i[0]
            .reshape([-1, 1])
            .dot(i[1].reshape([1, -1]))
            .reshape(i[0].shape.concat(i[1].shape))
            .sum(sum)
            .reshape([-1])
            .gather(tf.range(0, stop, step, "int32"))
            .reshape([-1])
            .sum([])

    })
}