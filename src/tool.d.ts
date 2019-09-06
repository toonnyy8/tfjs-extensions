import * as tf from "@tensorflow/tfjs"

declare class SequenceTidy {
    funcs: [()=>{}]
    constructor(func: ()=>{})
    next(func: ()=>{}):SequenceTidy
    run(input:any):tf.Tensor 
}

export declare function sequenceTidy (func: ()=>{}):SequenceTidy

export declare class TensorPtr {
    _ptr:tf.Tensor
    ptr :tf.Tensor
    constructor()
}

export declare function tensorPtr():TensorPtr