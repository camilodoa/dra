/*
Feedforward Artificial Neural Network
*/

var fann = class {
    constructor() {
        /*
        Input shape is an array with a single number
        This network has one hidden layer
        Output shape is an array of 4 classes
        */
       // I/O shapes
       this.inputShape = [1, 1];
       this.outputShape = [4, 1];
       // Number of hidden neurons
       this.numHidden = 10;
       // First layer's outgoing weights and biases
       this.w1 = this.randn(this.inputShape[1], this.numHidden);
       this.b1 = this.randn(this.numHidden, 1);
       // Second layer's outgoing weights and biases
       this.w2 = this.randn(this.numHidden, this.outputShape[1]);
       this.b2 = this.randn(this.outputShape[1], 1);
    }
    step = function(x) {

    }
    // Activations
    sigmoid = function(x) {
        /*
        Sigmoid activation for an array of elements
        */
        sigmoidFunction = function (t) {return 1 / (1 + Math.exp(-t))}
        return this.recursiveMap(x, sigmoidFunction);
    }
    // Utility
    recursiveMap = function(arr, func) {
        // Base case 1
        if (arr.length === 0) return [];
        // Base case 2
        // If the array isn't holding more nested arrays, map normally
        if (arr.length > 0 && !Array.isArray(arr[0])) return arr.map(func);
        // Otherwise, recursively map nested arrays
        return arr.map(x => recursiveMap(x, func));
    }
     randn = function(i, j) {
        // Initialize random matrix of shape [i, j]
        let arr = [];
        for (let first = 0; first < i; first++) {
            let nestedArr = []
            for (let second = 0; second < j; second++) {
                nestedArr.push(Math.random());
            }
            arr.push(nestedArr);
        }
        return arr;
    }
}