import * as tf from "@tensorflow/tfjs"

declare class saveTensor {
    constructor(tensor: tf.Tensor)
    shape: Array<Number>
    dtype: 'float32' | 'int32' | 'bool' | 'complex64' | 'string'
    values: Array<String> | ArrayBuffer
    saveToChar: Boolean
}

declare class saveVariable extends saveTensor {
    constructor(variable: tf.Variable)
    name: String
    trainable: Boolean
}

declare function save(tensorList: tf.Tensor | Array<tf.Tensor>, saveToChar?: Boolean): Array<saveVariable> | Array<saveTensor>
declare function load(saveTensorList: tf.Tensor | Array<tf.Tensor>): Array<tf.Variable> | Array<tf.Tensor>
