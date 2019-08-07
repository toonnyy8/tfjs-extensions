import * as tf from "@tensorflow/tfjs"

export function repeatElements(x, rep, axis) {
    let reps = [...Array(x.shape.length)].map(_ => 1)
    reps[axis] = rep
    return tf.tile(x, reps)
}

export function batchDot(x, y, axis) {
    return tf.sum(tf.mul(x, y), axis)
}

export function einsum(equation = "", ...inputs) {
    let [inputShapes, outputShape] = equation.split('->')
    let inputShape = inputShapes.split(",")
    let axis = {}
    for (let i = 0; i < equation.length; i++) {
        axis[equation[i]] = equation[i] == "," || equation[i] == "-" || equation[i] == ">" || equation[i] == " " || equation[i] == "." ? undefined : 0
    }
}