import * as tf from "@tensorflow/tfjs"

export class saveTensor {
    constructor(t, saveToChar = false) {
        this.shape = t.shape
        this.dtype = t.dtype
        this.values = t.dataSync()
        this.saveToChar = saveToChar
        if (this.saveToChar) {
            this.values = [...new Int16Array(this.values.buffer)].map((v) => String.fromCharCode(v))
        }
    }
}

export class saveVariable extends saveTensor {
    constructor(t, saveToChar = false) {
        super(t, saveToChar)
        this.name = t.name
        this.trainable = t.trainable
    }
}

export function save(tList, saveToChar) {
    let save_ = (t) => {
        if (t instanceof tf.Tensor) {
            if (t instanceof tf.Variable) {
                return new saveVariable(t, saveToChar)
            } else {
                return new saveTensor(t, saveToChar)
            }
        } else {
            console.error(`tensor must be an instance of tf.Tensor`)
        }
    }

    if (tList instanceof Array) {
        return tList.map((t) => save_(t))
    } else {
        return [save_(tList)]
    }
}

export function load(sTList) {
    return tf.tidy(() => {
        let load_ = (sT) => {
            return tf.tidy(() => {
                if (sT instanceof saveTensor) {
                    let values
                    if (sT.saveToChar) {
                        values = new Int32Array(sT.values.map((v) => (v).charCodeAt(0)))
                    } else {
                        values = sT.values
                    }
                    let t = tf.tensor(values, sT.shape, sT.dtype)
                    if (sT instanceof saveVariable) {
                        return tf.variable(t, sT.trainable, sT.name)
                    } else {
                        return t
                    }
                } else {
                    console.error(`tensor must be an instance of saveTensor`)
                }
            })
        }

        if (sTList instanceof Array) {
            return sTList.map((t) => load_(t))
        } else {
            return [load_(sTList)]
        }
    })
}