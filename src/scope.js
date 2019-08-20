import * as tf from "@tensorflow/tfjs"

export class VariableScope {
    constructor(name) {
        this.scopeName = name == undefined ? "" : name

        if (Object.keys(VariableScope.scopeList).find((scopeListName) => this.scopeName == scopeListName) == undefined) {
            this._variableList = {}
            VariableScope.scopeList[this.scopeName] = this
        }

        return VariableScope.scopeList[this.scopeName]
    }

    with(f = (scope = this, scopeName = `${this.scopeName}/`) => { }) {
        f(this, `${this.scopeName}/`)
    }

    variableScope(name) {
        return new VariableScope(`${this.scopeName}/${name}`)
    }

    getVariable(name, shape, trainable = true, dtype = "float32") {
        if (Object.keys(this._variableList).find((variableListName) => name == variableListName) == undefined) {
            this._variableList[name] = tf.tidy(() => tf.variable(tf.randomNormal(shape).cast(dtype), trainable, `${this.scopeName}/${name}`, dtype))
        }
        return this._variableList[name]
    }

}
VariableScope.scopeList = {}


export function variableScope(name) {
    return new VariableScope(name)
}
