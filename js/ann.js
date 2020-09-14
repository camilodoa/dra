/*
Feedforward Artificial Neural Network
*/

var ann = class {
    constructor(regularizationLambda = 0.005, alpha = 0.001) {
        /*
        Input shape is an array with a single number
        This network has one hidden layer
        Output shape is an array of a single class
        */
       // I/O shapes
       this.inputShape = [1, 1]; // Only take in distance to goal
       this.outputShape = [1, 1]; // Reward
       // Number of hidden neurons
       this.numHidden = 10;
       // First layer's incoming weights and biases
       this.w1 = math.map(math.zeros(this.inputShape[1], this.numHidden), x => Math.random() / Math.sqrt(this.inputShape[1]));
       this.b1 = math.map(math.zeros(this.numHidden, 1), x => Math.random());
       // First layer's outgoing weights and biases
       this.w2 = math.map(math.zeros(this.numHidden, this.outputShape[1]), x => Math.random() / Math.sqrt(this.numHidden));
       this.b2 = math.map(math.zeros(this.outputShape[1], 1), x => Math.random());
       // Parameters
       this.regularizationLambda = regularizationLambda;
       this.alpha = alpha;
       // Input accumulation for batch learning
       this.batch = [];
       this.loss = 0;
    }
    step = function(input, reward) {
        // Step forward in time
        // Collect observables into a batch of inputs
        // When batch is filled, update network weights
        if (batch.length > 32) {
            this.train();
            batch = [[[input], [reward]]];
        } else {
            batch.push([[input], [reward]]);
        }
    }
    train = function() {
        // Use accumulated batch of inputs to train the network
        this.batch.forEach(input => {
            let x = math.reshape(math.matrix(input[0]), this.inputShape);
            let y = math.reshape(math.matrix(input[1]), this.outputShape);
            // Feedforward pass
            let a1 = x;
            let a2 = math.map(math.add(math.multiply(a1, this.w1), math.transpose(this.b1)), t => 1 / (1 + Math.exp(-t)));
            let a3 = math.map(math.add(math.multiply(a2, this.w2), math.transpose(this.b2)), t => 1 / (1 + Math.exp(-t)));
            // Backprop
            // (a3 - y) * a3 (1 - a3)
            let delta3 = math.dotMultiply(math.subtract(a3, y), math.map(a3, x => x * (1 - x)));
            // (delta3 dot w2.T) * a2 (1 - a2)
            let delta2 = math.dotMultiply(math.multiply(delta3, math.transpose(this.w2)), math.map(a2, x => x * (1- x)));
            // Delta weights
            let deltaW2 = math.multiply(math.transpose(a2), delta3);
            // let deltaW2 = this.dot(a2, delta3);
            let deltaB2 = a3;
            // let deltaB2 = a3;
            let deltaW1 = math.multiply(math.transpose(a1), delta2);
            // let deltaW1 = this.dot(a1, delta2);
            let deltaB1 = delta2;
            // Update
            let m = x._size[0];
            // w1 += -alpha * ((1 / m * deltaW1) + regularizationLambda * w1)
            this.w1 = math.add(this.w1, 
                math.add(
                    math.multiply(-this.alpha, math.multiply(1 / m, deltaW1)),
                    math.multiply(this.regularizationLambda, this.w1)
                )
            );
            // b1 += - alpha * (1 / m * deltaB1)
            this.b1 = math.add(this.b1, math.multiply(-this.alpha, math.multiply(1 / m, math.transpose(deltaB1))));
            // w2 += -alpha * ((1 / m * deltaW2) + regularizationLambda * w2)
            this.w2 = math.add(this.w2, 
                math.add(
                    math.multiply(-this.alpha, math.multiply(1 / m, deltaW2)),
                    math.multiply(this.regularizationLambda, this.w2)
                )
            );
            // b2 += -alpha * (1 / m * deltaB2)
            this.b2 = math.add(this.b2, math.multiply(-this.alpha, math.multiply(1 / m, math.transpose(deltaB2))));
            this.loss = this.cost(a3, y);
        }); 
        return this.loss;
    }
    predict = function(x) {
        // Predict the reward of state x
        // Add the dot product of the input and the first weight layer and the bias term
        // Take the sigmoid of that
        x = math.reshape(math.matrix(x), this.inputShape)
        let z1 = math.map(math.add(math.multiply(x, this.w1), math.transpose(this.b1)), t => 1 / (1 + Math.exp(-t)));
        // Repeat
        let z2 = math.map(math.add(math.multiply(z1, this.w2), math.transpose(this.b2)), t => 1 / (1 + Math.exp(-t)));
        return z2;
    }
    cost = function(out, y) {
        // MSE
        let loss = math.mean(math.square(math.subtract(out, y)));
        // L2 regularization
        // Square all the weights in each layer and sum them up
        // This step works to reduce overfitting by reducing weights
        loss += this.regularizationLambda * (math.sum(math.square(this.w1)) + math.sum(math.square(this.w2)));
        return loss;
    }
}

let network = new ann();
let prediction = network.predict([10]);
console.log(prediction);
network.batch = [[[10], [-10]]];
let loss = network.train();
console.log(loss);
