# Machine learning for meteroid parameter inversion

## Goal

The overall goal of the neural network project is two-fold: given meteoroid observation data, quickly find the physical parameters that correspond to the observation, and find the uncertainties for each parameter. This can be formulated in the following way,

$$F^{-1}(\bold{y}(t))=\bold{\hat{x}}$$

however, what $F$, $\bold{y}(t)$ and $\bold{x}$ actually represent is still undecided. The loss function is also not clear.

## Approaches to solving the problem

For the analysis, we attempted to solve two different levels of the problem. One level has no erosion and no wakes, and using only the physical parameters:

* initial mass $m_0$
* initial velocity $v_0$
* zenith angle $\varphi$
* density $\rho$
* ablation coefficient $\sigma_{ab}$

The second level has erosion but still no wakes and the only parameters are:

* the ones from the no erosion case
* erosion height $h_{er}$
* erosion coefficient $\sigma_{er}$
* mass index $s$
* minimum fragment mass $m_{min}$
* maximum fragment mass $m_{max}$.

For training, random physical parameters were produced and the simulation was generated. All neural networks are trained on data derived from the random physical parameters and the clean simulation data.

For most of the following approaches, the loss function of the neural network is quite simple, mean square error between the neural network output and the expected output. This must be a part of the loss function, otherwise the neural network will struggle to converge. However it is also possible, in the cases where only the physical parameters are compared, that the loss function can include the similarity of the corresponding simulations so that if the simulation is accurate while the physical parameters aren't, the neural network is still rewarded. This method requires an accurate and fast forward model to approximate the simulation which I haven't yet found (since it's prohibitively time consuming to run the actual simulation each time), but it's potentially close to being done. I think having this implemented will aid in the fitting, but I don't think that this will solve all the problems.

### Appoach 0: Original non-machine learning approach

The manual approach to this problem is purely trial-and-error, guess at a set of parameters, run the simulation and check if it works. Alternatively, an iterative algorithm can be used to find this set of parameters, or at least further optimize the given set of parameters.

In this method, particle swarm optimization can be used to find parameters, and for those parameters, the simulation is run and the result is compared to the data. This will give the global minimum, however doing so is extremely time consuming for even a single meteroid observation. The simulation can take seconds per set of parameters in the worse case, and particle swarm optimization requires checking a very large amount of sets of parameters to get the global minimum. As such this method is inpractical.

### Approach 1: CNN mapping $\bold{y}$ array to $\bold{x}$

This approach involves converting $\bold{y}(t)$ to a 2d array, where one dimension is the discretization of time and the other are the values at the index, time, height, magnitude, and distance.

Using various convolutional layers followed by densely interconnected layers performs well on some physical parameters but worse on others.

* It can predict initial mass, initial velocity and zenith angle
* It struggles with predicting density and ablation coefficient. It often predicts the values to be larger than they should be, but to a maximum. Though it performs better on ablation coefficient
* It can somewhat accurately detect erosion height
* It struggles with erosion coefficient
* It assumes mass index, minimum mass and maximum mass are constants, and thus completely fails on the parameters.

The convolutional layers have difficulty fully taking into account the erosion discontinuity, often internally smoothing it out. If this approach were to be used to fit clean data, this will have to be addressed.

### Approach 2: Using an autoencoder with convolutional layers

An autoencoder can be used to understand the latent space behind our simulated data. The autoencoder would map the input of approach 1 to a small set of parameters (I used 20), then map it back to the original input. Here I used convolutional layers in the same way of approach 1 to get the set of parameters, then when mapping the parameters to the output, I used dense layers followed by convolutional layers (I tried using a convolutional transpose but it yielded worse results, at least with the padding I was using).

With this set of parameters, I can apply principal component analysis to make them independent of each other. This results in a small set of parameters that can (mostly) uniquely identify meteoroid data. I can also use these parameters as a measurement of how "similar" two observations were, using the euclidean distance between the pca of the set of parameters.

I also attempted to train a densely interconnected neural network to map the set of physical parameters, however this yielded the same results as approach 1 (likely due to the similar structure between the two of them).

When experimenting with the decoding part of the autoencoder, I noticed that it struggled with the discontinuity in the derivative when erosion occurs. The autoencoder couldn't properly encode this nuance, suggesting that the convolutional layer itself cannot properly analyze this. This is something that should find a solution, since it will allow for more confident analysis in the erosion case, not just the non-erosion case.

### Approach 3: Using a neural network to speed up simulation (forward model) and using particle swarm optimization

Another approach is to do a speed improvement for approach 0, which suffers from being too time consuming. We can do this by using a neural network to approximate the simulation, leading to a large speedup due to taking a consistent time for any input parameters, allowing gpu speedup, and with neural networks in general being faster to compute than a simulation.

The reason for this approach being able to work is that the forward problem can be solved quite accurately with a neural network, which is not the case for the inverse problem.

This big challenge for this approach is to make an approximation to the simulation that is sufficiently accurate for any given set of parameters, including erosion parameters.

This method can still make use of the same strategy for calculating error (mentioned in the error characterization section), since the jacobian of the forward model can still be calculated, although it will be the partial derivative of the input with respect to the output as supposed to the other way around.

However, after testing out this approach, it is still quite time consuming. This is entirely due to the particle swarm method requiring a lot of time to find the global minimum in such a high dimensional space. Despite this, if you are willing to wait, this method can get a quite accurate fit.

#### Approach 3.1: Using an array for y

This approach can be done with $y$ formulated as an array similar to approach 1. Here, we would input an array of physical parameters and the neural network, constructed using convolutational and dense layers, would calculate all $y$ values at once.

The problem with this approach is the same as what's suggested in approach 1 and 2, a convolutional layer struggles to reproduce the discontinuity in derivative when erosion occurs, thus it struggles to approximate the simulation.

#### Approach 3.2: Using a y value with a corresponding time

Another approach for the forward problem is, instead of outputting an array corresponding to $y$, the neural network can input all parameters and a height value, then output the corresponding $y$ value at that height.

This approach has the benefit of calculating $y$ only for height values that matter. As well, this neural network can be constructed with entirely dense layers and calculates only one height at a time, so it lends itself better to feature engineering.

Without feature engineering, the neural network outputs $y$ values that are much smoother than approaches 3.1 or the decoder from approach 2. However, in this case, it still struggles with the discontinuity when erosion occurs. To combat this, an extra feature can be added, a boolean that represents whether erosion has already occured at the given height. This gives a discontuity in the inputs, and thus results in a discintuity in the output. Overrall resulting in the desired discontinuity in derivative when erosion occurs.

With this improvement, the neural network can quite accurately approximates the no-erosion case, but still struggles a bit after erosion occurs. This can be due to a lack of training in this area in addition to the inherent difficulty of erosion. This is definitely an avenue for further investigation.

### Approach 4: Physics-informed neural network (PINN) for forward and inverse problem

#### Approach 4.1: Forward problem

Another approach to this problem is to use a physics-informed neural network (PINN) to approximate the forward problem, and to use its functionality to solve the inverse problem. A PINN can do anything that approach 3.2 can do except it uses extra information in the loss function to make sure that it always outputs something physical. It does so by incorporting ordinary or partial differential equations and initial or boundary conditions into the loss function.

To have this approach solve the general forward problem, I used every physical parameter as a variable, similar to the height. Doing method makes it very similar to approach 3.2, and just acts as a way to improve its accuracy.

The difficulty of this approach is that for the erosion case, there isn't a closed form system of differential equations to use as of now. This is because there are differential equations that apply to single body meteoroids, but with fragmentation, you have to apply those odes to each fragment, which may have different velocities, height and mass. An option for solving this problem is to instead solve a system of odes for a single parameter $\eta(v, h,m, t)$ which represents the density/quantity of meteoroids with the given properties at a given time.

Doing this approach requires training data, as used for all previous neural networks. While it can be done without, the neural network struggles to converge on an optimal solution.

Doing approach could be quite useful, since it's an extension on approach 3.2 without any tradeoffs. However, this approach will take some work to implement and may not provide much of an improvement on approach 3.2.

#### Approach 4.2: Inverse problem

The PINN also has the ability to solve the inverse problem, however it does so in a way that is quite limiting. In order to make use of this, the physical parameters cannot be inputs to the neural network, leaving only height as the input and $y$ as the output. Then the neural network is trained on a single measurement, and it finds the physical parameters which make the data best satisfy the system of odes. This can take a large amount of time to optimize on each measurement, and it doesn't offer uncertainties in the physical parameters.

While at first glace, a neural network that can solve the inverse problem may be exactly what we're looking for, the limitations of this approach make it not very desirable.

### Approach 5: Fitting curves to data and mapping fit parameters to physical parameters

The final approach that was considered was to fit a curve to the data and using a neural network to map the fit parameters to the corresponding physical parameters. For the distance, $x_0+vt+ae^{-bt}$ was fit and for magnitude the beta distribution was fit.

One benefit to this method is that with the `scipy` library, `scipy.optimize.curve_fit` is capable of fitting data with uncertainties and ouputting fit parameters with a corresponding covariance matrix. This covariance matrix could then easily be used in the calculation of physical parameter uncertainties.

The problem here is that the beta distribution has five degrees of freedom, and it only fits to the no-erosion case. For the erosion case, even more parameters will have to be fit, potentially totalling more than 10 parameters. This is a large amount of degrees of freedom, which will likely lead to the fit failing when little magnitude data is given.

As well, when this approach was used on simulated data in the no-erosion case (which doesn't have the issue of too little data to fit to), the neural network ran into the same difficulties as approach 1 and 2, and only some parameters could be properly predicted based on the fit parameters.

It is also worth mentioning that when attempting the forward problem for this data, that is, mapping physical parameters to fit parameters, the neural network actually does worse than the inverse problem. This is contrary to all the other forward problem approaches that have been done.

## Visualization

There is a recurring problem with solving the inverse problem: certain parameters can be fit properly, but others cannot. In the no-erosion case, the density and ablation coefficient are fit badly for any direct approach to the inverse problem. Due to the corresponding forward problem fitting quite well in comparison, the problem with density and ablation coefficient should be looked into.

The method for visualizing the accuracy of the neural network is to plot the correlations between the expected/correct physical parameters and the predicted physical parameters as outputted by the neural network. My method of visualizing this was to make a heatmap (as supposed to a scatter plot) where all predicted physical parameters are plotted against all correct physical parameters. Along the diagonal, you want straight lines and any of these plots which aren't straight lines were improperly fitted. Everywhere else, the variables should be indepedent, so you would expect each plot to be uniformly distributed.

Another primary tool for visualization is the autoencoder from approach 2, which is especially precise in the no-erosion case. Here the neural network and principal component analysis is able to assemble all data into a few important independent parameters $\bold{a}(\bold{x})$. $\bold{a}$ can then be used to measure similiarity between different sets of parameters $\bold{x}$. For instance, the similarity between two sets of parameters can be defined as follows:
$$S(\bold{x}_1, \bold{x}_2)=||\bold{a}(\bold{x}_1)-\bold{a}(\bold{x}_2)||^2$$
This parameter can also be used to measure sensitivity of the output on $\bold{x}$. One way that this can be forumated is as follows:
$$ D(\bold{x})=\sqrt{\sum_{ij}\left(\frac{\partial a_i}{\partial x_j}\right)^2}$$
Another visualization is one that measures fit accuracy, that is, the how well the inverse model performs on different physical parameters $\bold{x}$. Suppose we have an encoder part of an autoencoder which maps magnitude and velocity data to a small set of parameters. Mapping these parameters to the physical parameters is the inverse problem, and mapping the physical parameters to these parameters is the forward problem. The forward model $F$ is able to be fit quite accurately, but not the inverse model $F^{-1}$. So we can measure fit accuracy of physical parameter in the following way:
$$ \bold{A}(\bold{x})=(F^{-1}(F(\bold{x}))-\bold{x})^2$$
where the exponenta A is applied element-wise.

Each of the three above metrics can be visualized in the gui that I developed that can visualize high dimensional space by plotting a heatmap with two dimensions at a time. However, even with this visualization done, it's not clear to me exactly how this visualization corresponds to a bad fit.

A method that I attempted to interpret this was to train a neural network using generated data (using the forward model that takes physical parameters and outputs a set of parameters that came from the autoencoder) using samples from different regions of the physical parameter space. It could be the case that regions where many $\bold{x}$ values give similar simulations cause the neural network to have a hard time fitting some parameters and that's what's causing the neural network to struggle on the entire domain. If this was the case, it could be possible to train a neural network only on regions that play nice, or train the neural network more on the difficult regions.

## Physical parameter uncertainties

The error associated with physical parameters can be found easily for certain approaches. If an approach uses a neural network to output physical parameters when inputting some values with known uncertainty, the error in the output can be calculated quickly and easily.

Suppose we have a neural network $F$ that takes in an input array $\bold{y}$ of some sort and outputs physical parameters $\bold{x}$. The error in $\bold{x}$ can be found with propagation of uncertainties:
$$s_{x_i}=\sqrt{\sum_{n} \left(\frac{\partial x_i}{\partial y_n}\right)^2 s_{y_n}}$$
where $s_{x_i}$ is the uncertainty in the $i\text{th}$ physical parameter and $s_{y_i}$ is the uncertainty of each value inputted into the neural network. The assumption here is that the parameters inputted into the neural network are independent of each other. Otherwise the covariance matrix would be required with the following equation
$$S_x=JS_yJ^\top$$
where $J$ is the jacobian and $S_x$ and $S_y$ are both covariance matrices. We can make this calculation because the jacobian can be calculated using the automatic differentiation feature of some machine learning libraries.

In order to make this calculation, and even for approaches that don't involve knowledge of the jacobian, we need uncertainties and possibly a covariance matrix meteoroid measurements. This can be found by characterizing the error in the magnitude and velocity plots, since these plots are the primary locations with noise. Both of these plots are non-linear functions of time and height so using a naive method for calculating the covariance matrix won't work. Instead, I fit a high order polynomial to the data to the magnitude and fit the equation $x+vt+ae^{-bt}$ to the distance, then compiled all the residuals. With these residuals, the variance can be calculated for different bins of velocity, magnitude and lag. Fitting a curve to the variance at each bin, we get a value for $s_y(v,M,d)$ at any desired point. Finally, we assume that due to the non-linear nature of magnitude and velocity, they are not correlated.

Another method for computing uncertainties is by using the similarity quantity $S(\bold{x}_1, \bold{x}_2)$ mentioned in the previous section. This quantity is useful since it can be used as a way to approximate the uncertainty in some $\bold{x}$ measurement, which may have been found manually. For instance, given some measurement $\bold{x}_0$, we can find a region of possible values which could also be solutions to the data due to each of the $\bold{x}$ values having similar outputs. Picking a reasonable constant $s$, we can find a region of $X$ where $S(\bold{x}_0, \bold{x})\leq s, \forall \bold{x}\in X$. Then we can easily find the covariance matrix for all $\bold{x}\in X$.
