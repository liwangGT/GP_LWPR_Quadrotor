
# coding: utf-8

# In[1]:


import GPy


# As of Mon 12th of Oct running on devel branch of GPy 0.8.8

# In[2]:


GPy.plotting.change_plotting_library('plotly')


# # Gaussian process regression tutorial
# 
# ### Nicolas Durrande 2013
# #### with edits by James Hensman and Neil D. Lawrence
# 
# We will see in this tutorial the basics for building a 1 dimensional and a 2 dimensional Gaussian process regression model, also known as a kriging model.
# 
# We first import the libraries we will need:

# In[3]:


import numpy as np


# ## 1-dimensional model
# 
# For this toy example, we assume we have the following inputs and outputs:

# In[4]:


X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X) + np.random.randn(20,1)*0.05


# Note that the observations Y include some noise.
# 
# The first step is to define the covariance kernel we want to use for the model. We choose here a kernel based on Gaussian kernel (i.e. rbf or square exponential):

# In[5]:


kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)


# The parameter input_dim stands for the dimension of the input space. The parameters `variance` and `lengthscale` are optional, and default to 1. Many other kernels are implemented, type `GPy.kern.<tab>` to see a list

# In[6]:


#type GPy.kern.<tab> here:
get_ipython().magic('pinfo GPy.kern.BasisFuncKernel')


# The inputs required for building the model are the observations and the kernel:

# In[7]:


m = GPy.models.GPRegression(X,Y,kernel)


# By default, some observation noise is added to the model. The functions `display` and `plot` give an insight of the model we have just built:

# In[8]:


from IPython.display import display
display(m)


# In[9]:


fig = m.plot()
GPy.plotting.show(fig, filename='basic_gp_regression_notebook')


# The above cell shows our GP regression model before optimization of the parameters. The shaded region corresponds to ~95% confidence intervals (ie +/- 2 standard deviation).
# 
# The default values of the kernel parameters may not be optimal for the current data (for example, the confidence intervals seems too wide on the previous figure). A common approach is to find the values of the parameters that maximize the likelihood of the data. It as easy as calling `m.optimize` in GPy:

# In[10]:


m.optimize(messages=True)


# If we want to perform some restarts to try to improve the result of the optimization, we can use the `optimize_restarts` function. This selects random (drawn from $N(0,1)$) initializations for the parameter values, optimizes each, and sets the model to the best solution found.

# In[11]:


m.optimize_restarts(num_restarts = 10)


# In this simple example, the objective function (usually!) has only one local minima, and each of the found solutions are the same. 
# 
# Once again, we can use `print(m)` and `m.plot()` to look at the resulting model resulting model. This time, the paraemters values have been optimized agains the log likelihood (aka the log marginal likelihood): the fit shoul dbe much better. 

# In[12]:


display(m)
fig = m.plot()
GPy.plotting.show(fig, filename='basic_gp_regression_notebook_optimized')


# ### New plotting of GPy 0.9 and later
# The new plotting allows you to plot the density of a GP object more fine grained by plotting more percentiles of the distribution color coded by their opacity

# In[13]:


display(m)
fig = m.plot(plot_density=True)
GPy.plotting.show(fig, filename='basic_gp_regression_density_notebook_optimized')


# ## 2-dimensional example
# 
# Here is a 2 dimensional example:

# In[14]:


# sample inputs and outputs
X = np.random.uniform(-3.,3.,(50,2))
Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(50,1)*0.05

# define kernel
ker = GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)

# create simple GP model
m = GPy.models.GPRegression(X,Y,ker)

# optimize and plot
m.optimize(messages=True,max_f_eval = 1000)
fig = m.plot()
display(GPy.plotting.show(fig, filename='basic_gp_regression_notebook_2d'))
display(m)


# The flag `ARD=True` in the definition of the `Matern` kernel specifies that we want one lengthscale parameter per dimension (ie the GP is not isotropic). Note that for 2-d plotting, only the mean is shown. 

# ## Plotting slices
# To see the uncertaintly associated with the above predictions, we can plot slices through the surface. this is done by passing the optional `fixed_inputs` argument to the plot function. `fixed_inputs` is a list of tuples containing which of the inputs to fix, and to which value.
# 
# To get horixontal slices of the above GP, we'll fix second (index 1) input to -1, 0, and 1.5:

# In[15]:


slices = [-1, 0, 1.5]
figure = GPy.plotting.plotting_library().figure(3, 1, 
                        shared_xaxes=True,
                        subplot_titles=('slice at -1', 
                                        'slice at 0', 
                                        'slice at 1.5', 
                                        )
                            )
for i, y in zip(range(3), slices):
    canvas = m.plot(figure=figure, fixed_inputs=[(1,y)], row=(i+1), plot_data=False)
GPy.plotting.show(canvas, filename='basic_gp_regression_notebook_slicing')


# A few things to note:
#  * we've also passed the optional `ax` argument, to mnake the GP plot on a particular subplot
#  * the data look strange here: we're seeing slices of the GP, but all the data are displayed, even though they might not be close to the current slice.

# To get vertical slices, we simply fixed the other input. We'll turn the display of data off also:

# In[16]:


slices = [-1, 0, 1.5]
figure = GPy.plotting.plotting_library().figure(3, 1, 
                        shared_xaxes=True,
                        subplot_titles=('slice at -1', 
                                        'slice at 0', 
                                        'slice at 1.5', 
                                        )
                            )
for i, y in zip(range(3), slices):
    canvas = m.plot(figure=figure, fixed_inputs=[(0,y)], row=(i+1), plot_data=False)
GPy.plotting.show(canvas, filename='basic_gp_regression_notebook_slicing_vertical')


# You can find a host of other plotting options in the `m.plot` docstring. `Type m.plot?<enter>` to see. 

