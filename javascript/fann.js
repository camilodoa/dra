/*
Feedforward Artificial Neural Network
*/

var FANN = class {
    constructor(regLambda = 0.005) {
        /*
        Input shape is an array with a single number
        This network has one hidden layer
        Output shape is an array of 4 classes
        */
       // I/O shapes
       let c = document.getElementById('space')
       let canvas = c.getContext('2d');
       // 1D array of pixel values
       let inputPixels = canvas.getImageData(0, 0, c.width, c.height).data;
       this.inputShape = [1, inputPixels.length]; // Raw pixels (batch size, number of pixels)
       this.outputShape = [1, 1]; // Reward
       // Number of hidden neurons
       this.numHidden = 10;
       // First layer's outgoing weights and biases
       this.w1 = this.recursiveMap(this.randn(this.numHidden, this.inputShape[1]), x => x / Math.sqrt(this.inputShape[1]));
       this.b1 = this.randn(this.numHidden, 1);
       // Second layer's outgoing weights and biases
       this.w2 = this.recursiveMap(this.randn(this.outputShape[1], this.numHidden), x => x / Math.sqrt(this.numHidden));
       this.b2 = this.randn(this.outputShape[1], 1);
       // Parameters
       this.regularizationLambda = regLambda;
    }
    predict = function(x) {
        
    }
    cost = function(out, y) {
        // MSE
        let cost = this.mse(out, y);
        // L2 regularization
        // Square all the weights in each layer and sum them up
        // This step works to reduce overfitting by reducing weights
        let squaredSumW1 = this.recursiveSum(this.recursiveMap(this.w1, x => Math.pow(x, 2)))[0];
        let squaredSumW2 = this.recursiveSum(this.recursiveMap(this.w2, x => Math.pow(x, 2)))[0];
        cost += this.regularizationLambda *  (squaredSumW1 + squaredSumW2);
        return cost;
    }
    mse = function(out, y) {
        // Mean of the squared difference of predicted and actual
        let sum = this.recursiveSum(this.recursiveMap(this.recursiveSubtraction(out, y), x => Math.pow(x, 2)));
        // Corner case
        if (sum[1] === 0) return 0;
        return sum[0] / sum[1];
    }
    // Activations
    sigmoid = function(x) {
        // Sigmoid activation for an array of elements
        sigmoidFunction = function (t) {return 1 / (1 + Math.exp(-t))};
        return this.recursiveMap(x, sigmoidFunction);
    }
    // Utility
    recursiveMap = function(arr, func) {
        // Map, but it's safe for nested arrays
        // Base case 1
        if (arr.length === 0) return [];
        // Base case 2
        // If the array isn't holding more nested arrays, map normally
        if (arr.length > 0 && !Array.isArray(arr[0])) return arr.map(func);
        // Otherwise, recursively map nested arrays
        return arr.map(x => this.recursiveMap(x, func));
    }
    recursiveSubtraction = function(arr1, arr2) {
        // Recursively subtracts two arrays
        // Base cases
        // if either array is empty
        if (arr1.length === 0  || arr2.length === 0) return [];
        // if there are no more nested arrays
        if (arr1.length > 0 && !Array.isArray(arr1[0]) && arr2.length > 0 && !Array.isArray(arr2[0])) {
            // Element wise subtraction
            return arr1.map((e, i) => e - arr2[i]);
        }
        return arr1.map((e, i) => this.recursiveSubtraction(e, arr2[i]));
    }
    recursiveMult = function(arr1, arr2) {
        // Recursively multiplies two arrays
        // Base cases
        // if either array is empty
        if (arr1.length === 0  || arr2.length === 0) return [];
        // if there are no more nested arrays
        if (arr1.length > 0 && !Array.isArray(arr1[0]) && arr2.length > 0 && !Array.isArray(arr2[0])) {
            // Element wise subtraction
            return arr1.map((e, i) => e * arr2[i]);
        }
        return arr1.map((e, i) => this.recursiveMult(e, arr2[i]));
    }
    recursiveSum = function(arr) {
        // Returns array of total sum, and number of elements
        // Base cases
        if (arr.length === 0) return [0, 0];
        if (arr.length > 0 && !Array.isArray(arr[0])) return [arr.reduce((acc, e) => acc + e), arr.length];
        let ret = [0, 0];
        arr.forEach(e => {
            let values = this.recursiveSum(e);
            ret[0] += values[0];
            ret[1] += values[1];
        });
        return ret;
    }
    dot = function(arr1, arr2) {
        let larger = Array.isArray(arr1[0]) ? arr1 : arr2;
        let smaller = Array.isArray(arr1[0]) ? arr2 : arr1;
        console.log(larger);
        console.log(smaller);

        let result = []
        for (let i = 0; i < larger.length; i ++) {
            // Multiply each value in the element array of the larger matrix by the element in the smaller one
            let currSum = larger[i].reduce((acc, curr, i) => acc + (curr * smaller[i]));
            result.push(currSum);
        }
        console.log(result)
        return result;
        
        // Dot product
        // return this.recursiveSum(this.recursiveMult(arr1, arr2))[0];
    }
    transpose = function(mat) {
        // Transpose matrix
        // Copy matrix
        let arr = [...mat];
        // Transpose 2D array
        for (let i = 0; i < arr.length; i++) {
            for (let j = 0; j < i; j++) {
                const temp = arr[i][j];
                arr[i][j] = arr[j][i];
                arr[j][i] = temp;
            }
        }
        return arr
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

let network = new FANN();
console.log(network.w1);
// dot product test
let c = document.getElementById('space')
let canvas = c.getContext('2d');
// 1D array of pixel values
let inputPixels = canvas.getImageData(0, 0, c.width, c.height).data;
let z1 = network.dot(canvas.getImageData(0, 0, c.width, c.height).data, network.w1);
console.log(z1);
console.log(network.dot(z1, network.w2));