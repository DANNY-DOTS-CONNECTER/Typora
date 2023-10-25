[TOC]

## Basic

- state $s$
- action $a$
- policy $\pi$ 条件概率函数  $\pi(left|s)=0.2$   动作action是<u>随机</u>的
- RL靠奖励reward来学习，奖励非常重要 
- 状态转移函数state transition，下一个状态是<u>随机</u> （来源于环境，例如怪物的移动是随机的）$p(s'|s,a)$

<img src="./RL assets/image-20230520102607773.png" alt="image-20230520102607773" style="zoom:25%;" />

(state, action, reward)

随机性的两个来源：

- action

  Given state s, the action can be random（根据$\pi$随机抽样得到的）

- 状态转移：环境根据状态转移函数$p$来抽样的



- Return：回报，**未来的累计奖励**$U_t = R_t + R_{t+1} + ...$

- Discounted return$U_t$：折扣回报，未来奖励越来越不重要， $\gamma$是折扣率
  
- $U_t=R_t+\gamma R_{t+1} + \gamma^2 R_{t+2}+...$   是一个随机变量（不是一个随机数）
  
  - 奖励$R_i$取决于$S_i$和$A_i$，当前状态和当前动作有关。所以$U_t$和后面所有状态和所有动作都有关系，所以是随机的。
  - Note: $E(R_{t+1})=0.7\times0.8\times R^1+0.2\times 0.8\times R^2+...$
  
- **动作**价值函数action-value function $Q(s,a)$

  $Q_\pi (s_t,a_t)=E[U_t|S_t = s_t,A_t=a_t]$

  本质上是求期望。意义就是：如果用policy function $\pi$，那么在$s_t$状态下，做这个动作$a_t$是好还是坏，能够得到的期望得分是多少。会打分，就知道哪个动作好哪个动作不好了。（因为叫”**动作**价值函数嘛“）
  
  - 为什么要求期望呢？
  
    因为在t时刻我们并不知道后面的动作和状态，所以我们用积分的形式确定，对$U_t$求期望就可以得到一个数记作$Q_{\pi}$; 注意积分后，除了$s_t$和$a_t$<u>其余动作函数都积分积掉了</u>，所以$Q_{\pi}$只和$s_t$和$a_t$有关(他们俩都作为观测到的数值对待，而不是随机变量)，还有$\pi$ 有关。
  
  最优动作价值函数optimal action-value function：$Q^*=\underset{\pi}max Q_ {\pi}$   在$s_t$状态下做了动作$a_t$，最好的得分就是$Q^*$，和$\pi$无关。
  
  意思就是：我们有多种policy函数$\pi$，我们采用最好的那种policy函数。
  
- **状态**价值函数state-value function $V_\pi(s_t)=E_A[Q_\pi(s_t,A)]$  
  
  $0.7\times 向上动作价值函数 + 0.2\times 向左动作价值函数 + 0.1 \times 向右动作价值函数$
  
  在$s_t$状态下对动作求期望，把A给去掉，得到的就是策略$\pi$
  
  <img src="./RL assets/image-20230520111050863.png" alt="image-20230520111050863" style="zoom: 25%;" />
  
  求期望，只和$\pi, s_t$有关，与动作无关。可以直观告诉我们当前的**situation**如何，我们是快赢了还是快输了。也可以评价policy的好坏，如果$\pi$越好，那么$V_{\pi}$的平均值就越大。
  
  



两种学习思路：

- **基于策略的学习**

  根据固定策略$\pi$，随机选择动作$a_t$

- **基于价值学习**

  根据$Q^*$做动作。每观察到一个状态就做动作，让$Q^*$评价动作，选择最大值的动作
  
  <u>上面两个都是根据输入$s_t$，输出动作$a_t$</u>

## 基于价值学习

就是要学习一个函数来近似$Q^*$，方法就是DQN

### Deep Q-Network（DQN）

用神经网络近似$Q^*$。把$Q^*$当作先知，选择平均回报最多的动作。

但实际上我们没有$Q^*$，价值学习的想法就是学习一个想法来近似$Q^*$。 我们把这个<u>神经网络记作$Q(s,a;w)$</u>

- 输入是状态$s$，输出是很多数值$q_t$，数值是对每一个动作的打分。通过奖励来学习神经网络，打分就会越来越精确。

选择$a^*=argmaxQ^*(s,a)$  

### Temporal Difference (TD) Learning

怎么训练DQN呢？最常用的方法是TD Learning，通过奖励来更新参数。  

相当于，我们不需要打完游戏就能更新参数。打一部分就可以了；

在等式的右边有一部分是<u>真实</u>的奖励，另一部分是DQN在$t+1$时刻做的<u>估计</u>。

<img src="./RL assets/image-20230520134915961.png" alt="image-20230520134915961" style="zoom:25%;" />



相邻折扣回报之间的关系：$U_{t}=R_t+\gamma \times U_{t+1}$

等式左边是对$U_t$做出的估计，右边是对$U_{t+1}$的估计

左边是prediction，右边的作为<u>TD target</u>

<img src="./RL assets/image-20230520135303741.png" alt="image-20230520135303741" style="zoom:25%;" />

我们希望Prediction尽量接近TD target，所以用梯度下降法减小loss，<u>更新模型参数$w$</u>。

$w_t$是模型参数，每个$a_{t+1}$就是DQN选择的最大的a

<img src="./RL assets/image-20230520140106001.png" alt="image-20230520140106001" style="zoom: 33%;" />

## 基于策略学习

<u>神经网络policy network近似策略函数</u>

Policy function    $\pi(a|s)$

用<u>策略网络</u> $\pi(a|s;\theta)$来近似策略函数 $\pi(a|s)$，$\theta$是随机初始化的神经网络参数

<img src="./RL assets/image-20230520140545967.png" alt="image-20230520140545967" style="zoom:25%;" />

softmax激活函数，这样输出的是三维向量，每个值是动作对应的概率

$\sum_{a\in A}\pi(a|s;\theta) = 1$，这就是为什么要softmax层，它可以让加和等于1，然后输出的都是正数

### 状态价值函数近似

把$V_\pi(s_t)$中替换一下变成了$V_\pi(s_t;\theta)$

<img src="./RL assets/image-20230520142557327.png" alt="image-20230520142557327" style="zoom:25%;" />

<img src="./RL assets/image-20230520142606572.png" alt="image-20230520142606572" style="zoom:25%;" />

策略网络越好，$V$的值就越大，所以我们定义期望$J(\theta)=E_s[V(S;\theta)]$，将S作为随机变量，用期望给去掉，这样变量只剩下$\theta$，目标就是让这个<u>越大越好</u>

用策略梯度<u>上升</u>算法

### 策略梯度Policy gradient来学习$\theta$

- 梯度用蒙特卡洛进行近似，因为算不出来。也就是用$g(\hat a,\theta)$来近似$\frac{\partial V(s;\theta)}{\partial \theta}$。别管数学原理（用随机样本来近似期望）

#### 算法

<img src="./RL assets/image-20230520143754996.png" alt="image-20230520143754996" style="zoom:25%;" />

2. 随机抽样得到$a_t$
3. 计算价值函数$q_t$
4. 对策略函数求导，算出导数。$d_{\theta,t}$大小和$\theta$一样。
5. 近似算策略梯度$g(a_t,\theta_t)$，也就是梯度的<u>蒙特卡洛近似</u>
6. 有了梯度，就可以更新策略网络的参数了。做梯度上升，目的是让状态价值函数$V$变大

- $q_t$是算不出来的，然后$q_t$可以用神经网络来近似，引出actor-critic方法

### summary

- 我们需要一个策略函数，所以我们用一个policy network去近似这个函数。这个network怎么学习呢？通过策略梯度学习。

- 策略梯度是价值函数$V$关于$\theta$的导数，也就是最大化$E_s[V(S;\theta)]$。算出策略梯度，根据梯度上升来更新$\theta$。

## actor-critic 方法

<u>把价值学习和策略学习结合起来。</u>

根据状态价值函数，分别学习$\pi$和$Q_{\pi}$

策略网络对应actor，可以看作是<u>运动员</u>。决策是由policy network做的，近似：$\pi(a|s;\theta)$

价值网络对应critic，可以看作是<u>裁判</u>。价值网络value network来近似：$q(s,a;w)$，不控制动作，只是打分

<img src="./RL assets/image-20230520145518082.png" alt="image-20230520145518082" style="zoom:25%;" />

<u>Policy Network:</u> 

- 策略网络控制运动

<img src="./RL assets/image-20230520145724073.png" alt="image-20230520145724073" style="zoom:25%;" />

<u>Value network:</u>

- 

<img src="./RL assets/image-20230520145647950.png" alt="image-20230520145647950" style="zoom:25%;" />

### Train

- update policy network $\pi(a|s;\theta)$ 是为了<u>增加状态价值</u>$V$。$V$越大说明策略$\pi$越好。**监督**来自于价值网络<u>critic</u>，逐渐表演得更好。

- update value network $q(s,a;w)$ 是为了更好估计<u>未来奖励的总和return</u>。一开始打分都是乱打的

  **监督**来自环境给的<u>奖励</u>，是为了让打分return更精准（裁判要做的就是逐渐接近上帝的打分）

### 算法  

<img src="./RL assets/image-20230520150321252.png" alt="image-20230520150321252" style="zoom:25%;" />

actor：策略网络用策略梯度算法来训练；目的是让**$V$的平均值变大**，这个期望很难求出来，用$g(A,\theta)$就是对这个期望（$E_A[g(A,\theta)]$）的 <u>蒙特卡洛近似</u>。$g(A,\theta)$是对策略梯度的<u>无偏估计</u>，所以直接**代替**策略梯度，梯度就做一次梯度上升，运动员的平均分就提高了。

critic：价值网络用TD 算法来训练，用了$q_t$和TD target（都是对价值回报的估计），用$y_t$来估计更靠谱，求损失Loss函数，然后做梯度下降，让$q_t更接近y_t$，让裁判打分更精准。

 <img src="./RL assets/image-20231018224052712.png" alt="image-20231018224052712" style="zoom: 67%;" /> 

#### 整体算法

<img src="./RL assets/image-20231018225647928.png" alt="image-20231018224052712" style="zoom: 67%;" />

最后一步可以用TD error $\delta_t$代替$q_t$作为baseline，收敛会更快

## AlphaGo

棋盘是19*19的，一共有361个点。

训练步骤：

- behavior cloning

  模仿人类行为，监督学习，初步学习策略网络

- 强化学习进一步训练，策略梯度算法，自我博弈。

- 训练价值网络

### Policy Network

#### 架构

## Monte Carlo Algorithms

蒙特卡洛计算期望

## Sarsa

TD算法的一种



因为$y_t$由$r_t$和上一轮得出的$q(s_{t+1},a_{t+1};w)$组成，而$r_t$与$w$无关，且$q(s_{t+1},a_{t+1};w)$中的$w$为上一轮已更新过的参数，为常数，也与$w$无关
