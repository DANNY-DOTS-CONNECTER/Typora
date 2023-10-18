- state $s$

- action $a$

- policy $\pi$ 条件概率函数  $\pi(left|s)=0.2$   动作action是<u>随机</u>的

- RL靠奖励reward来学习，奖励非常重要 

- 状态转移函数state transition，下一个状态是<u>随机</u> （来源于环境，例如怪物的移动是随机的）$p(s'|s,a)$

<img src="./RL assets/image-20230520102551640.png" alt="image-20230520102551640" style="zoom:25%;" />

<img src="./RL assets/image-20230520102607773.png" alt="image-20230520102607773" style="zoom:25%;" />

(state, action, reward)

- Return：回报，未来的累计奖励$U_t$

- Discounted return：折扣回报，未来奖励越来越不重要$U_t=R_t+\gamma R_{t+1} + \gamma^2 R_{t+2}+...$   是一个随机变量（不是一个随机数）
  - 奖励$R_i$取决于$S_i$和$A_i$，当前状态和当前动作有关。所以$U_t$和后面所有状态和所有动作都有关系，所以是随机的。
  - Note: $E(R_{t+1})=0.7\times0.8\times R^1+0.2\times 0.8\times R^2+...$

- 动作价值函数action-value function $Q(s,a)$

  $Q_\pi (s_t,a_t)=E[U_t|S_t = s_t,A_t=a_t]$

  本质上是求期望
  
  最优动作价值函数：$Q^*=maxQ_\pi$   在$s_t$状态下做了动作$a_t$，最好的得分就是$Q^*$，和$\pi$无关
  
- 状态价值函数state-value function $V_\pi(s_t)=E_A[Q_\pi(s_t,A)]$  
  
  $0.7\times 向上动作价值函数 + 0.2\times 向左动作价值函数 + 0.1 \times 向右动作价值函数$
  
  在$s_t$状态下对动作求期望，得到的就是策略$\pi$
  
  <img src="./RL assets/image-20230520111050863.png" alt="image-20230520111050863" style="zoom: 25%;" />
  
- 动作价值函数：$\pi,s_t,a_t$都有关系，在当前状态st下做了动作at，将来仍然使用策略$\pi$，能够得到的期望得分是多少。（和后面的$s_{t+1},s_{t+2}...以及a_{t+1},a_{t+2}...无关$）

  状态价值函数：$\pi,s_t$有关，在动作空间中对动作价值函数求期望，给定策略，判断当前处境好不好，<u>和动作无关</u>（积分积掉了）

  $Q^*$和$\pi$无关，相当于评分。



两种学习思路：

- 基于策略的学习

  根据策略随机选择动作

- 基于价值学习

  根据$Q^*$，选择最大值的策略

## 基于价值学习

就是要学习一个函数来近似$Q^*$，方法就是DQN

### Deep Q-Network

选择$a^*=argmaxQ^*(s,a)$  

用DQN拟合$Q^*$，这里面只有状态转移函数的随机性。输出的是一个向量

### Temporal Difference (TD) Learning

怎么训练DQN呢？最常用的方法是TD Learning，通过奖励来更新参数。

用梯度下降法降低TD error

<img src="./RL assets/image-20230520134915961.png" alt="image-20230520134915961" style="zoom:25%;" />

$U_{t}=R_t+\gamma \times U_{t+1}$

<img src="./RL assets/image-20230520135303741.png" alt="image-20230520135303741" style="zoom:25%;" />

$w_t$是模型参数，每个$a_{t+1}$就是DQN选择的最大的a

减少Loss

<img src="./RL assets/image-20230520140106001.png" alt="image-20230520140106001" style="zoom: 33%;" />

## 基于策略学习

神经网络近似策略函数

Policy function    $\pi(a|s)$

用<u>策略网络</u>$\pi(a|s;\theta)$来近似策略函数，$\theta$是随机初始化的参数

<img src="./RL assets/image-20230520140545967.png" alt="image-20230520140545967" style="zoom:25%;" />

softmax激活函数，这样输出的是三维向量，每个值是动作对应的概率

$\sum_{a\in A}\pi(a|s;\theta) = 1$，这就是为什么要softmax层，它可以让加和等于1，然后输出的都是正数

### 状态价值函数近似

$U_t$ 的随机性来自于所有未来的状态

把$V_\pi(s_t)$中替换一下变成了$V_\pi(s_t;\theta)$

<img src="./RL assets/image-20230520142557327.png" alt="image-20230520142557327" style="zoom:25%;" />

<img src="./RL assets/image-20230520142606572.png" alt="image-20230520142606572" style="zoom:25%;" />

策略网络越好，$V$的值就越大，所以我们定义期望$J(\theta)=E_s[V(S;\theta)]$，目标就是让这个越大越好

用策略梯度上升算法

#### 策略梯度Policy gradient来学习$\theta$

梯度用蒙特卡洛进行近似

<img src="./RL assets/image-20230520143754996.png" alt="image-20230520143754996" style="zoom:25%;" />

然后$q_t$可以用神经网络来近似，引出actor-critic方法

## actor-critic 方法

策略网络对应actor，决策是由策略网络做的，近似：$\pi(a|s;\theta)$

价值网络对应critic，近似：$q(s,a;w)$，不控制动作，只是打分

<img src="./RL assets/image-20230520145518082.png" alt="image-20230520145518082" style="zoom:25%;" />

Policy Network: 

- 监督来自于价值网络critic，逐渐表演得更好

<img src="./RL assets/image-20230520145724073.png" alt="image-20230520145724073" style="zoom:25%;" />

Value network:

- 监督来自环境给的<u>奖励</u>，是为了让打分return更精准（裁判要做的就是逐渐接近上帝的打分）

<img src="./RL assets/image-20230520145647950.png" alt="image-20230520145647950" style="zoom:25%;" />

  

<img src="./RL assets/image-20230520150321252.png" alt="image-20230520150321252" style="zoom:25%;" />

价值网络用TD 算法来训练，用了$q_t$和TD target（都是对价值回报的估计），用$y_t$来估计更靠谱，求损失Loss函数，然后做梯度下降，让$q_t更接近y_t$

策略网络用策略梯度算法来训练；目的是让$V$的平均值变大，用蒙特卡洛近似来算这个梯度。有了梯度就做一次梯度上升，运动员的平均分就提高了。

