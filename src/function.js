import * as tf from "@tensorflow/tfjs"

export function repeatElements(x, rep, axis) {
    let reps = [...Array(x.shape.length)].map(_ => 1)
    reps[axis] = rep
    return tf.tile(x, reps)
}

export function batchDot(x, y, axis) {
    return tf.sum(tf.mul(x, y), axis)
}

export function mergeShape(tensor, axises) {
    return tf.tidy(() => {
        let shape = tensor.shape
        let merge = -1
        let newShape = []
        if (axises) {
            shape.forEach((val, idx) => {
                if (axises.find(axis => axis == idx) != undefined) {
                    if (merge == -1) {
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
        return tensor.reshape(newShape || shape)
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
        console.log(Math.max(...arr[idx]))
        arr[idx].forEach(() => {

        })
    })
    console.log(equation)


    // let [inputShapes, outputShape] = equation.split('->')
    // let inputShape = inputShapes.split(",")
    // let axis = {}
    // for (let i = 0; i < equation.length; i++) {
    //     axis[equation[i]] = equation[i] == "," || equation[i] == "-" || equation[i] == ">" || equation[i] == " " || equation[i] == "." ? undefined : 0
    // }
}