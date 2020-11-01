/*
camilo ortiz, 2020, camilodoa.ml
feed forward artificial neural network
dependencies:
math.js
*/

var ffnn = class {
    constructor(regularizationLambda = 0.005, alpha = 0.001, numHidden = 10, batchSize = 32) {
        /*
        Input shape is an array with a single number
        This network has one hidden layer
        Output shape is an array of a single class
        */
       // I/O shapes
       this.inputShape = [1, 1]; // Only take in distance to goal
       this.outputShape = [1, 1]; // Reward
       // Number of hidden neurons
       this.numHidden = numHidden;
       // First layer's incoming weights and biases
       this.w1 = math.map(math.zeros(this.inputShape[1], this.numHidden), function(x) {
           return Math.random() / Math.sqrt(this.inputShape[1]);
       }.bind(this));
       this.b1 = math.map(math.zeros(this.numHidden, 1), function(x) {
           return Math.random();
       });
       // First layer's outgoing weights and biases
       this.w2 = math.map(math.zeros(this.numHidden, this.outputShape[1]), function(x) {
           return Math.random() / Math.sqrt(this.numHidden);
       }.bind(this));
       this.b2 = math.map(math.zeros(this.outputShape[1], 1), function(x) {
           return Math.random();
       });
       // Parameters
       this.regularizationLambda = regularizationLambda;
       this.alpha = alpha;
       this.batchSize = batchSize;
       // Input accumulation for batch learning
       this.batch = [];
       this.loss = 0;
    }
    step = function(input, reward) {
        // Step forward in time
        // Collect observables into a batch of inputs
        // When batch is filled, update network weights
        if (this.batch.length > this.batchSize) {
            this.train();
            this.batch = [[[input], [reward]]];
        } else {
            this.batch.push([[input], [reward]]);
        }
    }
    train = function() {
        // Use accumulated batch of inputs to train the network
        this.batch.forEach(input => {
            var x = math.reshape(math.matrix(input[0]), this.inputShape);
            var y = math.reshape(math.matrix(input[1]), this.outputShape);
            // Feedforward pass
            var a1 = x;
            var a2 = math.map(math.add(math.multiply(a1, this.w1), math.transpose(this.b1)), this.sigmoid);
            var a3 = math.map(math.add(math.multiply(a2, this.w2), math.transpose(this.b2)), this.sigmoid);
            // Backprop
            // (a3 - y) * a3 (1 - a3)
            var delta3 = math.dotMultiply(math.subtract(a3, y), math.map(a3, function(x) {
                return x * (1 - x);
            }));
            // (delta3 dot w2.T) * a2 (1 - a2)
            var delta2 = math.dotMultiply(math.multiply(delta3, math.transpose(this.w2)), math.map(a2, function(x) {
                return x * (1- x);
            }));
            // Delta weights
            var deltaW2 = math.multiply(math.transpose(a2), delta3);
            var deltaB2 = a3;
            var deltaW1 = math.multiply(math.transpose(a1), delta2);
            var deltaB1 = delta2;
            // Update
            var m = x._size[0];
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
        var z1 = math.map(math.add(math.multiply(x, this.w1), math.transpose(this.b1)), this.sigmoid);
        // Repeat
        var z2 = math.map(math.add(math.multiply(z1, this.w2), math.transpose(this.b2)), this.sigmoid);
        return z2;
    }
    see = function(x) {
      space = document.getElementById("space").getContext("2d").getImageData();
    }
    cost = function(out, y) {
        // MSE
        var loss = math.mean(math.square(math.subtract(out, y)));
        // L2 regularization
        // Square all the weights in each layer and sum them up
        // This step works to reduce overfitting by reducing weights
        loss += this.regularizationLambda * (math.sum(math.square(this.w1)) + math.sum(math.square(this.w2)));
        return loss;
    }
    sigmoid = function(t) {
        return 1 / (1 + Math.exp(-t));
    }
}
