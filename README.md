conda create --name <env> --file requirements.txt

# RESULTS

## Escaping the Maze : Fixed Time Horizon using Dynamic Programming

<img src="Results/Problem1_Policy_GIF.gif" width="50%" height="50%" ></img><img src="Results/Problem1_plot.png" width="50%" height="50%" ></img>

##  Escaping the Maze : Infinite time horizon : 
1. 

<table>
  <tr>
    <td>MDP-Value Iteration</td>
     <td>Model free-Qlearning</td>
     <td>Optimal Policy</td>
  </tr>
  <tr>
    <td><img src="Results/Q2_7.png"></img></td>
    <td><img src="Results/Q2_1.png"></img></td>
    <td><img src="Results/Q2_2.png"></img></td>
  </tr>
 </table>

##  Escaping the police : Infinite time horizon :

<table>
  <tr>
    <td>Q learning </td>
     <td>SARSA with epsilon greedy</td>
     <td>Optimal Policy</td>
   
  </tr>
  <tr>
    <td><img src="Results/Q3_1.png"></img></td>
    <td><img src="Results/Q3_3.png"></img></td>
    <td><img src="Results/Q3_2.png"></img></td>
  </tr>
 </table>


##  Mountain Car : SARSA with Eligibility Traces (Linear Function Approximations using Fourier basis)

<table>
  <tr>
    <td>Reward over episodes</td>
     <td>Cost to go</td>
   
  </tr>
  <tr>
    <td><img src="Results/Q4_1.png"></img></td>
    <td><img src="Results/Q4_2.png"></img></td>
   
  </tr>
 </table>