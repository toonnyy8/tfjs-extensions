import * as tf from "@tensorflow/tfjs"

class SequenceTidy {
    constructor(startFunc = () => {}) {
        this.funcs = []
        this.funcs.push(startFunc)
    }

    next(func = () => {}) {
        this.funcs.push(func)
        return this
    }

    run(...inputs) {
        return tf.tidy(() => {
            let output = [tf.tidy(() => {
                return this.funcs[0](...inputs)
            })]
            for (let i = 1; i < this.funcs.length; i++) {
                let newOutput = [tf.tidy(() => {
                    return this.funcs[i](...output)
                })]
                tf.dispose(output)
                output = newOutput
            }
            return output
        })
    }
}

export function sequenceTidy(func = () => {}) {
    return new SequenceTidy(func)
}

export class TensorPtr {
    constructor() {
        this._ptr = null
    }
    get ptr() {
        return this._ptr
    }
    set ptr(value) {
        tf.dispose(this._ptr)
        this._ptr = value
    }
}

export function tensorPtr() {
    return new TensorPtr()
}