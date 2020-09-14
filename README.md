# Deep Reinforcement Learning Agent [(Dra)](https://camilodoa.ml/dra)

![image](img/image.png)

Dra is a "deep" reinforcement learning agent that exists in your browser. The reward at every time
step is negatively correlated with Dra's distance to the goal. The training network is making 
predictions of the given reward at each time step given the current Euclidean distance to the goal.

Here are the current parameters:

```javascript
this.regularizationLambda = 0.005;
this.alpha = 0.001;
this.batchSize = 1;
this.numHidden = 10; // Single hidden layer with 10 neurons
```

When you open the page, Dra starts updating a fundamental rule set of its
environment. When you close the page, it forgets.

Made with `paper.js`.
Inspired by [otoro.net](https://otoro.net/).
