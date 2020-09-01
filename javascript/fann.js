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
     randn = function(i, j) {
        // Initialize random matrix of shape [i, j]
        let arr = [];
        for (let i = 0; i < 1; i ++) {
            let nestedArr = []
            for (let j = 0; j < this.numHidden; j++) {
                nestedArr.push(Math.random());
            }
            arr.push(nestedArr);
        }
        return arr;
    }
}