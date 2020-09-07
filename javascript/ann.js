/*
Feedforward Artificial Neural Network
*/

var ann = class {
    constructor(regularizationLambda = 0.005, alpha = 0.001, sight = 1000) {
        /*
        Input shape is an array with a single number
        This network has one hidden layer
        Output shape is an array of 4 classes
        */
       // I/O shapes
       let c = document.getElementById('space')
       let canvas = c.getContext('2d');
       // 1D array of pixel values
    //    let inputPixels = canvas.getImageData(position.x - sight, position.y - sight, position.x + sight, position.y + sight).data;
       let inputPixels = canvas.getImageData(0, 0, c.clientWidth, c.clientHeight);
       console.log(inputPixels)
       this.inputShape = [32, inputPixels.length]; // Raw pixels (batch size, number of pixels)
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
       this.regularizationLambda = regularizationLambda;
       this.alpha = alpha;
       // Input accumulation for batch learning
       this.batch = [];
    }
    step = function(input, reward) {
        // Step forward in time
        // Collect observables into a batch of inputs
        // When batch is filled, update network weights
        if (batch.length > 32) {
            this.train();
            batch = [[input, [reward]]];
        } else {
            batch.push([input, [reward]]);
        }
    }
    train = function() {
        // Use accumulated batch of inputs to train the network
        this.batch.forEach(input => {
            let x = input[0];
            let y = input[1];
            // Feedforward pass
            let a1 = x;
            let a2 = this.sigmoid(this.recursiveSumUp(this.dot(x, this.w1), this.b1));
            let a3 = this.sigmoid(this.recursiveSumUp(this.dot(a1, this.w2), this.b2));
            // Backprop
            // (out - y) * a3 (1 - a3)
            let delta3 = this.recursiveMultiplication(this.recursiveSubtraction(a3, y), this.recursiveMultiplication(a3, this.recursiveMap(a3, x => 1 - x)));
            // (delta3 dot w2.T) * a2 (1 - a2)
            let delta2 = this.recursiveMultiplication(this.dot(delta3, this.w2), this.recursiveMultiplication(a2, this.recursiveMap(a2, x => 1 - x)));
            // Delta weights
            let deltaW2 = this.dot(a2, delta3);
            let deltaB2 = a3;
            let deltaW1 = this.dot(a1, delta2);
            let deltaB1 = delta2;
            // Update
            let m = x.length;
            // w1 += -alpha * ((1 / m * deltaW1) + regularizationLambda * w1)
            this.w1 = this.recursiveSum(self.w1, this.recursiveMap(
                this.recursiveSum(
                    this.recursiveMap(deltaW1, x => (1 / m) * x), 
                    this.recursiveMap(this.w1, x => x * this.regularizationLambda)
                ), x => x * - this.alpha
            ));
            // b1 += - alpha * (1 / m * deltaB1)
            this.b1 = this.recursiveSum(self.b1, this.recursiveMap(
                this.recursiveMap(deltaB1, x => (1 / m) * x),
                x => x * - this.alpha
            ));
            // w2 += -alpha * ((1 / m * deltaW2) + regularizationLambda * w2)
            this.w2 = this.recursiveSum(self.w2, this.recursiveMap(
                this.recursiveSum(
                    this.recursiveMap(deltaW2, x => (1 / m) * x), 
                    this.recursiveMap(this.w2, x => x * this.regularizationLambda)
                ), x => x * - this.alpha
            ));
            // b2 += -alpha * (1 / m * deltaB2)
            this.b2 = this.recursiveSum(self.b2, this.recursiveMap(
                this.recursiveMap(deltaB2, x => (1 / m) * x),
                x => x * - this.alpha
            ));
            console.log(this.cost(this.predict(x), y))
        }) 
    }
    predict = function(x) {
        // Predict the reward of state x
        // Add the dot product of the input and the first weight layer and the bias term
        // Take the sigmoid of that
        let z1 = this.sigmoid(this.recursiveSumUp(this.dot(x, this.w1), this.b1));
        // Repeat
        let z2 = this.sigmoid(this.recursiveSumUp(this.dot(z1, this.w2), this.b2));
        return z2;
    }
    cost = function(out, y) {
        // MSE
        let cost = this.mse(out, y);
        // L2 regularization
        // Square all the weights in each layer and sum them up
        // This step works to reduce overfitting by reducing weights
        let squaredSumW1 = this.recursiveSumUp(this.recursiveMap(this.w1, x => Math.pow(x, 2)))[0];
        let squaredSumW2 = this.recursiveSumUp(this.recursiveMap(this.w2, x => Math.pow(x, 2)))[0];
        cost += this.regularizationLambda *  (squaredSumW1 + squaredSumW2);
        return cost;
    }
    mse = function(out, y) {
        // Mean of the squared difference of predicted and actual
        let sum = this.recursiveSumUp(this.recursiveMap(this.recursiveSubtraction(out, y), x => Math.pow(x, 2)));
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
    recursiveSum = function(arr1, arr2) {
        // Recursively sums two arrays, making a new array of the same shape
        // Base cases
        // if either array is empty
        if (arr1.length === 0  || arr2.length === 0) return [];
        // if there are no more nested arrays
        if (arr1.length > 0 && !Array.isArray(arr1[0]) && arr2.length > 0 && !Array.isArray(arr2[0])) {
            // Element wise subtraction
            return arr1.map((e, i) => e + arr2[i]);
        }
        return arr1.map((e, i) => this.recursiveSubtraction(e, arr2[i]));
    }
    recursiveMultiplication = function(arr1, arr2) {
        // Recursively multiplies two arrays
        // Base cases
        // if either array is empty
        if (arr1.length === 0  || arr2.length === 0) return [];
        // if there are no more nested arrays
        if (arr1.length > 0 && !Array.isArray(arr1[0]) && arr2.length > 0 && !Array.isArray(arr2[0])) {
            // Element wise subtraction
            return arr1.map((e, i) => e * arr2[i]);
        }
        return arr1.map((e, i) => this.recursiveMultiplication(e, arr2[i]));
    }
    recursiveSumUp = function(arr) {
        // Returns array of total sum, and number of elements
        // Base cases
        if (arr.length === 0) return [0, 0];
        if (arr.length > 0 && !Array.isArray(arr[0])) return [arr.reduce((acc, e) => acc + e), arr.length];
        let ret = [0, 0];
        arr.forEach(e => {
            let values = this.recursiveSumUp(e);
            ret[0] += values[0];
            ret[1] += values[1];
        });
        return ret;
    }
    dot = function(arr1, arr2) {
        // Dot product of 2 arrays
        // Only works if one array is 2D and the other is 1D
        // Probably should be fixed
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
        return result;
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
