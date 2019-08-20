import * as tf from "@tensorflow/tfjs"

export declare class VariableScope {
    scopeName: String
    constructor(name: String)
    with(func: (scope: VariableScope, scopeName: String) => tf.Tensor): tf.Tensor
    variableScope(name: String): VariableScope
    getVariable(name: String, shape?: Number[], trainable?: Boolean, dtype?: "float32" | "int32" | "bool" | "complex64" | "string"): tf.Variable
    dispose(name?: String): void
}

export declare function variableScope(name: String): VariableScope 