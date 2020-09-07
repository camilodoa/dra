# Deep Reinforcement Learning Agent [(Dra)](https://dra.rlitb.ml)

![image](image.png)

Dra is a deep reinforcement learning agent that exists in your browser. The reward at every time
step is negatively correlated with Dra's distance to the goal. The training network is making 
predictions of the given reward at each time step given the site's raw pixel values.

When you open the page, Dra starts updating a fundamental rule set of its
environment. When you close the page or resize it, it forgets.

Made with `paper.js`.
Inspired by [otoro.net](https://otoro.net/).
