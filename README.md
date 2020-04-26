# Inferential-Statistics 1-a
Frequentist Stats
Inferential Statistics Ia - Frequentism
Learning objectives
Welcome to the first Frequentist inference mini-project! Over the course of working on this mini-project and the next frequentist mini-project, you'll learn the fundamental concepts associated with frequentist inference. The following list includes the topics you will become familiar with as you work through these two mini-projects:

the z-statistic
the t-statistic
the difference and relationship between the two
the Central Limit Theorem, including its assumptions and consequences
how to estimate the population mean and standard deviation from a sample
the concept of a sampling distribution of a test statistic, particularly for the mean
how to combine these concepts to calculate a confidence interval
Prerequisites
For working through this notebook, you are expected to have a very basic understanding of:

what a random variable is
what a probability density function (pdf) is
what the cumulative density function is
a high-level sense of what the Normal distribution
If these concepts are new to you, please take a few moments to Google these topics in order to get a sense of what they are and how you might use them.

While it's great if you have previous knowledge about sampling distributions, this assignment will introduce the concept and set you up to practice working using sampling distributions. This notebook was designed to bridge the gap between having a basic understanding of probability and random variables and being able to apply these concepts in Python. The second frequentist inference mini-project focuses on a real-world application of this type of inference to give you further practice using these concepts.

For this notebook, we will use data sampled from a known normal distribution. This allows us to compare our results with theoretical expectations.

I An introduction to sampling from the Normal distribution
First, let's explore the ways we can generate the Normal distribution. While there's a fair amount of interest in sklearn within the machine learning community, you're likely to have heard of scipy if you're coming from the sciences. For this assignment, you'll use scipy.stats to complete your work.

In [18]:
from scipy.stats import norm
from scipy.stats import t
import numpy as np
import pandas as pd
from numpy.random import seed
import matplotlib.pyplot as plt
Q: Call up the documentation for the norm function imported above. What is the second listed method?

In [19]:
help(norm)
Help on norm_gen in module scipy.stats._continuous_distns object:

class norm_gen(scipy.stats._distn_infrastructure.rv_continuous)
 |  norm_gen(momtype=1, a=None, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, extradoc=None, seed=None)
 |  
 |  A normal continuous random variable.
 |  
 |  The location (``loc``) keyword specifies the mean.
 |  The scale (``scale``) keyword specifies the standard deviation.
 |  
 |  %(before_notes)s
 |  
 |  Notes
 |  -----
 |  The probability density function for `norm` is:
 |  
 |  .. math::
 |  
 |      f(x) = \frac{\exp(-x^2/2)}{\sqrt{2\pi}}
 |  
 |  for a real number :math:`x`.
 |  
 |  %(after_notes)s
 |  
 |  %(example)s
 |  
 |  Method resolution order:
 |      norm_gen
 |      scipy.stats._distn_infrastructure.rv_continuous
 |      scipy.stats._distn_infrastructure.rv_generic
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  fit(self, data, **kwds)
 |      Return MLEs for shape (if applicable), location, and scale
 |      parameters from data.
 |      
 |      MLE stands for Maximum Likelihood Estimate.  Starting estimates for
 |      the fit are given by input arguments; for any arguments not provided
 |      with starting estimates, ``self._fitstart(data)`` is called to generate
 |      such.
 |      
 |      One can hold some parameters fixed to specific values by passing in
 |      keyword arguments ``f0``, ``f1``, ..., ``fn`` (for shape parameters)
 |      and ``floc`` and ``fscale`` (for location and scale parameters,
 |      respectively).
 |      
 |      Parameters
 |      ----------
 |      data : array_like
 |          Data to use in calculating the MLEs.
 |      args : floats, optional
 |          Starting value(s) for any shape-characterizing arguments (those not
 |          provided will be determined by a call to ``_fitstart(data)``).
 |          No default value.
 |      kwds : floats, optional
 |          Starting values for the location and scale parameters; no default.
 |          Special keyword arguments are recognized as holding certain
 |          parameters fixed:
 |      
 |          - f0...fn : hold respective shape parameters fixed.
 |            Alternatively, shape parameters to fix can be specified by name.
 |            For example, if ``self.shapes == "a, b"``, ``fa``and ``fix_a``
 |            are equivalent to ``f0``, and ``fb`` and ``fix_b`` are
 |            equivalent to ``f1``.
 |      
 |          - floc : hold location parameter fixed to specified value.
 |      
 |          - fscale : hold scale parameter fixed to specified value.
 |      
 |          - optimizer : The optimizer to use.  The optimizer must take ``func``,
 |            and starting position as the first two arguments,
 |            plus ``args`` (for extra arguments to pass to the
 |            function to be optimized) and ``disp=0`` to suppress
 |            output as keyword arguments.
 |      
 |      Returns
 |      -------
 |      mle_tuple : tuple of floats
 |          MLEs for any shape parameters (if applicable), followed by those
 |          for location and scale. For most random variables, shape statistics
 |          will be returned, but there are exceptions (e.g. ``norm``).
 |      
 |      Notes
 |      -----
 |      This function uses explicit formulas for the maximum likelihood
 |      estimation of the normal distribution parameters, so the
 |      `optimizer` argument is ignored.
 |      
 |      Examples
 |      --------
 |      
 |      Generate some data to fit: draw random variates from the `beta`
 |      distribution
 |      
 |      >>> from scipy.stats import beta
 |      >>> a, b = 1., 2.
 |      >>> x = beta.rvs(a, b, size=1000)
 |      
 |      Now we can fit all four parameters (``a``, ``b``, ``loc`` and ``scale``):
 |      
 |      >>> a1, b1, loc1, scale1 = beta.fit(x)
 |      
 |      We can also use some prior knowledge about the dataset: let's keep
 |      ``loc`` and ``scale`` fixed:
 |      
 |      >>> a1, b1, loc1, scale1 = beta.fit(x, floc=0, fscale=1)
 |      >>> loc1, scale1
 |      (0, 1)
 |      
 |      We can also keep shape parameters fixed by using ``f``-keywords. To
 |      keep the zero-th shape parameter ``a`` equal 1, use ``f0=1`` or,
 |      equivalently, ``fa=1``:
 |      
 |      >>> a1, b1, loc1, scale1 = beta.fit(x, fa=1, floc=0, fscale=1)
 |      >>> a1
 |      1
 |      
 |      Not all distributions return estimates for the shape parameters.
 |      ``norm`` for example just returns estimates for location and scale:
 |      
 |      >>> from scipy.stats import norm
 |      >>> x = norm.rvs(a, b, size=1000, random_state=123)
 |      >>> loc1, scale1 = norm.fit(x)
 |      >>> loc1, scale1
 |      (0.92087172783841631, 2.0015750750324668)
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from scipy.stats._distn_infrastructure.rv_continuous:
 |  
 |  __init__(self, momtype=1, a=None, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, extradoc=None, seed=None)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  cdf(self, x, *args, **kwds)
 |      Cumulative distribution function of the given RV.
 |      
 |      Parameters
 |      ----------
 |      x : array_like
 |          quantiles
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information)
 |      loc : array_like, optional
 |          location parameter (default=0)
 |      scale : array_like, optional
 |          scale parameter (default=1)
 |      
 |      Returns
 |      -------
 |      cdf : ndarray
 |          Cumulative distribution function evaluated at `x`
 |  
 |  expect(self, func=None, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
 |      Calculate expected value of a function with respect to the
 |      distribution by numerical integration.
 |      
 |      The expected value of a function ``f(x)`` with respect to a
 |      distribution ``dist`` is defined as::
 |      
 |                  ub
 |          E[f(x)] = Integral(f(x) * dist.pdf(x)),
 |                  lb
 |      
 |      where ``ub`` and ``lb`` are arguments and ``x`` has the ``dist.pdf(x)`` 
 |      distribution. If the bounds ``lb`` and ``ub`` correspond to the 
 |      support of the distribution, e.g. ``[-inf, inf]`` in the default 
 |      case, then the integral is the unrestricted expectation of ``f(x)``. 
 |      Also, the function ``f(x)`` may be defined such that ``f(x)`` is ``0`` 
 |      outside a finite interval in which case the expectation is 
 |      calculated within the finite range ``[lb, ub]``. 
 |      
 |      Parameters
 |      ----------
 |      func : callable, optional
 |          Function for which integral is calculated. Takes only one argument.
 |          The default is the identity mapping f(x) = x.
 |      args : tuple, optional
 |          Shape parameters of the distribution.
 |      loc : float, optional
 |          Location parameter (default=0).
 |      scale : float, optional
 |          Scale parameter (default=1).
 |      lb, ub : scalar, optional
 |          Lower and upper bound for integration. Default is set to the
 |          support of the distribution.
 |      conditional : bool, optional
 |          If True, the integral is corrected by the conditional probability
 |          of the integration interval.  The return value is the expectation
 |          of the function, conditional on being in the given interval.
 |          Default is False.
 |      
 |      Additional keyword arguments are passed to the integration routine.
 |      
 |      Returns
 |      -------
 |      expect : float
 |          The calculated expected value.
 |      
 |      Notes
 |      -----
 |      The integration behavior of this function is inherited from
 |      `scipy.integrate.quad`. Neither this function nor
 |      `scipy.integrate.quad` can verify whether the integral exists or is
 |      finite. For example ``cauchy(0).mean()`` returns ``np.nan`` and
 |      ``cauchy(0).expect()`` returns ``0.0``.
 |      
 |      Examples
 |      --------
 |      
 |      To understand the effect of the bounds of integration consider 
 |      >>> from scipy.stats import expon
 |      >>> expon(1).expect(lambda x: 1, lb=0.0, ub=2.0)
 |      0.6321205588285578
 |      
 |      This is close to 
 |      
 |      >>> expon(1).cdf(2.0) - expon(1).cdf(0.0)
 |      0.6321205588285577
 |      
 |      If ``conditional=True``
 |      
 |      >>> expon(1).expect(lambda x: 1, lb=0.0, ub=2.0, conditional=True)
 |      1.0000000000000002
 |      
 |      The slight deviation from 1 is due to numerical integration.
 |  
 |  fit_loc_scale(self, data, *args)
 |      Estimate loc and scale parameters from data using 1st and 2nd moments.
 |      
 |      Parameters
 |      ----------
 |      data : array_like
 |          Data to fit.
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information).
 |      
 |      Returns
 |      -------
 |      Lhat : float
 |          Estimated location parameter for the data.
 |      Shat : float
 |          Estimated scale parameter for the data.
 |  
 |  isf(self, q, *args, **kwds)
 |      Inverse survival function (inverse of `sf`) at q of the given RV.
 |      
 |      Parameters
 |      ----------
 |      q : array_like
 |          upper tail probability
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information)
 |      loc : array_like, optional
 |          location parameter (default=0)
 |      scale : array_like, optional
 |          scale parameter (default=1)
 |      
 |      Returns
 |      -------
 |      x : ndarray or scalar
 |          Quantile corresponding to the upper tail probability q.
 |  
 |  logcdf(self, x, *args, **kwds)
 |      Log of the cumulative distribution function at x of the given RV.
 |      
 |      Parameters
 |      ----------
 |      x : array_like
 |          quantiles
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information)
 |      loc : array_like, optional
 |          location parameter (default=0)
 |      scale : array_like, optional
 |          scale parameter (default=1)
 |      
 |      Returns
 |      -------
 |      logcdf : array_like
 |          Log of the cumulative distribution function evaluated at x
 |  
 |  logpdf(self, x, *args, **kwds)
 |      Log of the probability density function at x of the given RV.
 |      
 |      This uses a more numerically accurate calculation if available.
 |      
 |      Parameters
 |      ----------
 |      x : array_like
 |          quantiles
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information)
 |      loc : array_like, optional
 |          location parameter (default=0)
 |      scale : array_like, optional
 |          scale parameter (default=1)
 |      
 |      Returns
 |      -------
 |      logpdf : array_like
 |          Log of the probability density function evaluated at x
 |  
 |  logsf(self, x, *args, **kwds)
 |      Log of the survival function of the given RV.
 |      
 |      Returns the log of the "survival function," defined as (1 - `cdf`),
 |      evaluated at `x`.
 |      
 |      Parameters
 |      ----------
 |      x : array_like
 |          quantiles
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information)
 |      loc : array_like, optional
 |          location parameter (default=0)
 |      scale : array_like, optional
 |          scale parameter (default=1)
 |      
 |      Returns
 |      -------
 |      logsf : ndarray
 |          Log of the survival function evaluated at `x`.
 |  
 |  nnlf(self, theta, x)
 |      Return negative loglikelihood function.
 |      
 |      Notes
 |      -----
 |      This is ``-sum(log pdf(x, theta), axis=0)`` where `theta` are the
 |      parameters (including loc and scale).
 |  
 |  pdf(self, x, *args, **kwds)
 |      Probability density function at x of the given RV.
 |      
 |      Parameters
 |      ----------
 |      x : array_like
 |          quantiles
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information)
 |      loc : array_like, optional
 |          location parameter (default=0)
 |      scale : array_like, optional
 |          scale parameter (default=1)
 |      
 |      Returns
 |      -------
 |      pdf : ndarray
 |          Probability density function evaluated at x
 |  
 |  ppf(self, q, *args, **kwds)
 |      Percent point function (inverse of `cdf`) at q of the given RV.
 |      
 |      Parameters
 |      ----------
 |      q : array_like
 |          lower tail probability
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information)
 |      loc : array_like, optional
 |          location parameter (default=0)
 |      scale : array_like, optional
 |          scale parameter (default=1)
 |      
 |      Returns
 |      -------
 |      x : array_like
 |          quantile corresponding to the lower tail probability q.
 |  
 |  sf(self, x, *args, **kwds)
 |      Survival function (1 - `cdf`) at x of the given RV.
 |      
 |      Parameters
 |      ----------
 |      x : array_like
 |          quantiles
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information)
 |      loc : array_like, optional
 |          location parameter (default=0)
 |      scale : array_like, optional
 |          scale parameter (default=1)
 |      
 |      Returns
 |      -------
 |      sf : array_like
 |          Survival function evaluated at x
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from scipy.stats._distn_infrastructure.rv_generic:
 |  
 |  __call__(self, *args, **kwds)
 |      Freeze the distribution for the given arguments.
 |      
 |      Parameters
 |      ----------
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution.  Should include all
 |          the non-optional arguments, may include ``loc`` and ``scale``.
 |      
 |      Returns
 |      -------
 |      rv_frozen : rv_frozen instance
 |          The frozen distribution.
 |  
 |  __getstate__(self)
 |  
 |  __setstate__(self, state)
 |  
 |  entropy(self, *args, **kwds)
 |      Differential entropy of the RV.
 |      
 |      Parameters
 |      ----------
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information).
 |      loc : array_like, optional
 |          Location parameter (default=0).
 |      scale : array_like, optional  (continuous distributions only).
 |          Scale parameter (default=1).
 |      
 |      Notes
 |      -----
 |      Entropy is defined base `e`:
 |      
 |      >>> drv = rv_discrete(values=((0, 1), (0.5, 0.5)))
 |      >>> np.allclose(drv.entropy(), np.log(2.0))
 |      True
 |  
 |  freeze(self, *args, **kwds)
 |      Freeze the distribution for the given arguments.
 |      
 |      Parameters
 |      ----------
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution.  Should include all
 |          the non-optional arguments, may include ``loc`` and ``scale``.
 |      
 |      Returns
 |      -------
 |      rv_frozen : rv_frozen instance
 |          The frozen distribution.
 |  
 |  interval(self, alpha, *args, **kwds)
 |      Confidence interval with equal areas around the median.
 |      
 |      Parameters
 |      ----------
 |      alpha : array_like of float
 |          Probability that an rv will be drawn from the returned range.
 |          Each value should be in the range [0, 1].
 |      arg1, arg2, ... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information).
 |      loc : array_like, optional
 |          location parameter, Default is 0.
 |      scale : array_like, optional
 |          scale parameter, Default is 1.
 |      
 |      Returns
 |      -------
 |      a, b : ndarray of float
 |          end-points of range that contain ``100 * alpha %`` of the rv's
 |          possible values.
 |  
 |  mean(self, *args, **kwds)
 |      Mean of the distribution.
 |      
 |      Parameters
 |      ----------
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information)
 |      loc : array_like, optional
 |          location parameter (default=0)
 |      scale : array_like, optional
 |          scale parameter (default=1)
 |      
 |      Returns
 |      -------
 |      mean : float
 |          the mean of the distribution
 |  
 |  median(self, *args, **kwds)
 |      Median of the distribution.
 |      
 |      Parameters
 |      ----------
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information)
 |      loc : array_like, optional
 |          Location parameter, Default is 0.
 |      scale : array_like, optional
 |          Scale parameter, Default is 1.
 |      
 |      Returns
 |      -------
 |      median : float
 |          The median of the distribution.
 |      
 |      See Also
 |      --------
 |      stats.distributions.rv_discrete.ppf
 |          Inverse of the CDF
 |  
 |  moment(self, n, *args, **kwds)
 |      n-th order non-central moment of distribution.
 |      
 |      Parameters
 |      ----------
 |      n : int, n >= 1
 |          Order of moment.
 |      arg1, arg2, arg3,... : float
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information).
 |      loc : array_like, optional
 |          location parameter (default=0)
 |      scale : array_like, optional
 |          scale parameter (default=1)
 |  
 |  rvs(self, *args, **kwds)
 |      Random variates of given type.
 |      
 |      Parameters
 |      ----------
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information).
 |      loc : array_like, optional
 |          Location parameter (default=0).
 |      scale : array_like, optional
 |          Scale parameter (default=1).
 |      size : int or tuple of ints, optional
 |          Defining number of random variates (default is 1).
 |      random_state : None or int or ``np.random.RandomState`` instance, optional
 |          If int or RandomState, use it for drawing the random variates.
 |          If None, rely on ``self.random_state``.
 |          Default is None.
 |      
 |      Returns
 |      -------
 |      rvs : ndarray or scalar
 |          Random variates of given `size`.
 |  
 |  stats(self, *args, **kwds)
 |      Some statistics of the given RV.
 |      
 |      Parameters
 |      ----------
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information)
 |      loc : array_like, optional
 |          location parameter (default=0)
 |      scale : array_like, optional (continuous RVs only)
 |          scale parameter (default=1)
 |      moments : str, optional
 |          composed of letters ['mvsk'] defining which moments to compute:
 |          'm' = mean,
 |          'v' = variance,
 |          's' = (Fisher's) skew,
 |          'k' = (Fisher's) kurtosis.
 |          (default is 'mv')
 |      
 |      Returns
 |      -------
 |      stats : sequence
 |          of requested moments.
 |  
 |  std(self, *args, **kwds)
 |      Standard deviation of the distribution.
 |      
 |      Parameters
 |      ----------
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information)
 |      loc : array_like, optional
 |          location parameter (default=0)
 |      scale : array_like, optional
 |          scale parameter (default=1)
 |      
 |      Returns
 |      -------
 |      std : float
 |          standard deviation of the distribution
 |  
 |  var(self, *args, **kwds)
 |      Variance of the distribution.
 |      
 |      Parameters
 |      ----------
 |      arg1, arg2, arg3,... : array_like
 |          The shape parameter(s) for the distribution (see docstring of the
 |          instance object for more information)
 |      loc : array_like, optional
 |          location parameter (default=0)
 |      scale : array_like, optional
 |          scale parameter (default=1)
 |      
 |      Returns
 |      -------
 |      var : float
 |          the variance of the distribution
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from scipy.stats._distn_infrastructure.rv_generic:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  random_state
 |      Get or set the RandomState object for generating random variates.
 |      
 |      This can be either None or an existing RandomState object.
 |      
 |      If None (or np.random), use the RandomState singleton used by np.random.
 |      If already a RandomState instance, use it.
 |      If an int, use a new RandomState instance seeded with seed.

A:second method is cdf

Q: Use the method that generates random variates to draw five samples from the standard normal distribution.

A:rvs method is used

In [68]:
seed(47)
# draw five samples here
sample = norm.rvs(size=5)
sample
Out[68]:
array([-0.84800948,  1.30590636,  0.92420797,  0.6404118 , -1.05473698])
Q: What is the mean of this sample? Is it exactly equal to the value you expected? Hint: the sample was drawn from the standard normal distribution.

A:the meann is 0.193

In [21]:
# Calculate and print the mean here, hint: use np.mean()
mean = np.mean(sample)
mean
Out[21]:
0.19355593334131074
Q: What is the standard deviation of these numbers? Calculate this manually here as $\sqrt{\frac{\sum_i(x_i - \bar{x})^2}{n}}$. Hint: np.sqrt() and np.sum() will be useful here and remember that numpy supports broadcasting.

A:std is 0.96

In [22]:
np.sqrt( np.sum((sample - mean) ** 2)  /len(sample) )
Out[22]:
0.9606195639478641
Here we have calculated the actual standard deviation of a small (size 5) data set. But in this case, this small data set is actually a sample from our larger (infinite) population. In this case, the population is infinite because we could keep drawing our normal random variates until our computers die. In general, the sample mean we calculate will not be equal to the population mean (as we saw above). A consequence of this is that the sum of squares of the deviations from the population mean will be bigger than the sum of squares of the deviations from the sample mean. In other words, the sum of squares of the deviations from the sample mean is too small to give an unbiased estimate of the population variance. An example of this effect is given here. Scaling our estimate of the variance by the factor $n/(n-1)$ gives an unbiased estimator of the population variance. This factor is known as Bessel's correction. The consequence of this is that the $n$ in the denominator is replaced by $n-1$.

Q: If all we had to go on was our five samples, what would be our best estimate of the population standard deviation? Use Bessel's correction ($n-1$ in the denominator), thus $\sqrt{\frac{\sum_i(x_i - \bar{x})^2}{n-1}}$.

A:std is 1.07

In [25]:
np.sqrt( sum((sample - np.mean(sample)) ** 2)  /(len(sample) - 1))
Out[25]:
1.0740053227518152
Q: Now use numpy's std function to calculate the standard deviation of our random samples. Which of the above standard deviations did it return?

A:np.std()

In [26]:
np.std(sample)
Out[26]:
0.9606195639478641
Q: Consult the documentation for np.std() to see how to apply the correction for estimating the population parameter and verify this produces the expected result.

A:ddof parameter used for correction of population

In [27]:
help(np.std)
Help on function std in module numpy:

std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>)
    Compute the standard deviation along the specified axis.
    
    Returns the standard deviation, a measure of the spread of a distribution,
    of the array elements. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.
    
    Parameters
    ----------
    a : array_like
        Calculate the standard deviation of these values.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The
        default is to compute the standard deviation of the flattened array.
    
        .. versionadded:: 1.7.0
    
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it is
        the same as the array type.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
    
        If the default value is passed, then `keepdims` will not be
        passed through to the `std` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    
    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If `out` is None, return a new array containing the standard deviation,
        otherwise return a reference to the output array.
    
    See Also
    --------
    var, mean, nanmean, nanstd, nanvar
    numpy.doc.ufuncs : Section "Output arguments"
    
    Notes
    -----
    The standard deviation is the square root of the average of the squared
    deviations from the mean, i.e., ``std = sqrt(mean(abs(x - x.mean())**2))``.
    
    The average squared deviation is normally calculated as
    ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is specified,
    the divisor ``N - ddof`` is used instead. In standard statistical
    practice, ``ddof=1`` provides an unbiased estimator of the variance
    of the infinite population. ``ddof=0`` provides a maximum likelihood
    estimate of the variance for normally distributed variables. The
    standard deviation computed in this function is the square root of
    the estimated variance, so even with ``ddof=1``, it will not be an
    unbiased estimate of the standard deviation per se.
    
    Note that, for complex numbers, `std` takes the absolute
    value before squaring, so that the result is always real and nonnegative.
    
    For floating-point input, the *std* is computed using the same
    precision the input has. Depending on the input data, this can cause
    the results to be inaccurate, especially for float32 (see example below).
    Specifying a higher-accuracy accumulator using the `dtype` keyword can
    alleviate this issue.
    
    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.std(a)
    1.1180339887498949
    >>> np.std(a, axis=0)
    array([ 1.,  1.])
    >>> np.std(a, axis=1)
    array([ 0.5,  0.5])
    
    In single precision, std() can be inaccurate:
    
    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> np.std(a)
    0.45000005
    
    Computing the standard deviation in float64 is more accurate:
    
    >>> np.std(a, dtype=np.float64)
    0.44999999925494177

In [28]:
np.std(sample, ddof=1)
Out[28]:
1.0740053227518152
Summary of section
In this section, you've been introduced to the scipy.stats package and used it to draw a small sample from the standard normal distribution. You've calculated the average (the mean) of this sample and seen that this is not exactly equal to the expected population parameter (which we know because we're generating the random variates from a specific, known distribution). You've been introduced to two ways of calculating the standard deviation; one uses $n$ in the denominator and the other uses $n-1$ (Bessel's correction). You've also seen which of these calculations np.std() performs by default and how to get it to generate the other.

You use $n$ as the denominator if you want to calculate the standard deviation of a sequence of numbers. You use $n-1$ if you are using this sequence of numbers to estimate the population parameter. This brings us to some terminology that can be a little confusing.

The population parameter is traditionally written as $\sigma$ and the sample statistic as $s$. Rather unhelpfully, $s$ is also called the sample standard deviation (using $n-1$) whereas the standard deviation of the sample uses $n$. That's right, we have the sample standard deviation and the standard deviation of the sample and they're not the same thing!

The sample standard deviation$$
s = \sqrt{\frac{\sum_i(x_i - \bar{x})^2}{n-1}} \approx \sigma,
$$is our best (unbiased) estimate of the population parameter ($\sigma$).

If your data set is your entire population, you simply want to calculate the population parameter, $\sigma$, via$$
\sigma = \sqrt{\frac{\sum_i(x_i - \bar{x})^2}{n}}
$$as you have complete, full knowledge of your population. In other words, your sample is your population. It's worth noting at this point if your sample is your population then you know absolutely everything about your population, there are no probabilities really to calculate and no inference to be done.

If, however, you have sampled from your population, you only have partial knowledge of the state of your population and the standard deviation of your sample is not an unbiased estimate of the standard deviation of the population, in which case you seek to estimate that population parameter via the sample standard deviation, which uses the $n-1$ denominator.

You're now firmly in frequentist theory territory. Great work so far! Now let's dive deeper.

II Sampling distributions
So far we've been dealing with the concept of taking a sample from a population to infer the population parameters. One statistic we calculated for a sample was the mean. As our samples will be expected to vary from one draw to another, so will our sample statistics. If we were to perform repeat draws of size $n$ and calculate the mean of each, we would expect to obtain a distribution of values. This is the sampling distribution of the mean. The Central Limit Theorem (CLT) tells us that such a distribution will approach a normal distribution as $n$ increases. For the sampling distribution of the mean, the standard deviation of this distribution is given by

$$
\sigma_{mean} = \frac{\sigma}{\sqrt n}
$$
where $\sigma_{mean}$ is the standard deviation of the sampling distribution of the mean and $\sigma$ is the standard deviation of the population (the population parameter).

This is important because typically we are dealing with samples from populations and all we know about the population is what we see in the sample. From this sample, we want to make inferences about the population. We may do this, for example, by looking at the histogram of the values and by calculating the mean and standard deviation (as estimates of the population parameters), and so we are intrinsically interested in how these quantities vary across samples. In other words, now that we've taken one sample of size $n$ and made some claims about the general population, what if we were to take another sample of size $n$? Would we get the same result? Would we make the same claims about the general population? This brings us to a fundamental question: when we make some inference about a population based on our sample, how confident can we be that we've got it 'right'?

Let's give our normal distribution a little flavor. Also, for didactic purposes, the standard normal distribution, with its variance equal to its standard deviation of one, would not be a great illustration of a key point. Let us imagine we live in a town of 50000 people and we know the height of everyone in this town. We will have 50000 numbers that tell us everything about our population. We'll simulate these numbers now and put ourselves in one particular town, called 'town 47', where the population mean height is 172 cm and population standard deviation is 5 cm.

In [29]:
seed(47)
pop_heights = norm.rvs(172, 5, size=50000)
In [30]:
_ = plt.hist(pop_heights, bins=30)
_ = plt.xlabel('height (cm)')
_ = plt.ylabel('number of people')
_ = plt.title('Distribution of heights in entire town population')
_ = plt.axvline(172, color='r')
_ = plt.axvline(172+5, color='r', linestyle='--')
_ = plt.axvline(172-5, color='r', linestyle='--')
_ = plt.axvline(172+10, color='r', linestyle='-.')
_ = plt.axvline(172-10, color='r', linestyle='-.')

Now, 50000 people is rather a lot to chase after with a tape measure. If all you want to know is the average height of the townsfolk, then can you just go out and measure a sample to get a pretty good estimate of the average height?

In [31]:
def townsfolk_sampler(n):
    return np.random.choice(pop_heights, n)
Let's say you go out one day and randomly sample 10 people to measure.

In [32]:
seed(47)
daily_sample1 = townsfolk_sampler(10)
In [33]:
_ = plt.hist(daily_sample1, bins=10)
_ = plt.xlabel('height (cm)')
_ = plt.ylabel('number of people')
_ = plt.title('Distribution of heights in sample size 10')

The sample distribution doesn't look much like what we know (but wouldn't know in real-life) the population distribution looks like. What do we get for the mean?

In [35]:
np.std(daily_sample1)
Out[35]:
5.593371148771279
In [34]:
np.mean(daily_sample1)
Out[34]:
173.47911444163503
And if we went out and repeated this experiment?

In [10]:
daily_sample2 = townsfolk_sampler(10)
In [11]:
np.mean(daily_sample2)
Out[11]:
173.7317666636263
Q: Simulate performing this random trial every day for a year, calculating the mean of each daily sample of 10, and plot the resultant sampling distribution of the mean.

A:

In [45]:
sample_mean = []
In [46]:
seed(47)
# take your samples here
for i in range(365):
    daily_sample3 = townsfolk_sampler(10)
    sample_mean.append(np.mean(daily_sample3))
In [47]:
_ = plt.hist(sample_mean, bins=10)
_ = plt.xlabel('height (cm)')
_ = plt.ylabel('number of people')
_ = plt.title('Distribution of heights in sample size 10')

The above is the distribution of the means of samples of size 10 taken from our population. The Central Limit Theorem tells us the expected mean of this distribution will be equal to the population mean, and standard deviation will be $\sigma / \sqrt n$, which, in this case, should be approximately 1.58.

Q: Verify the above results from the CLT.

A:

In [52]:
np.std(sample_mean)
Out[52]:
1.5756704135286475
In [55]:
clt_std = 5 / np.sqrt(10)
clt_std
Out[55]:
1.5811388300841895
Remember, in this instance, we knew our population parameters, that the average height really is 172 cm and the standard deviation is 5 cm, and we see some of our daily estimates of the population mean were as low as around 168 and some as high as 176.

Q: Repeat the above year's worth of samples but for a sample size of 50 (perhaps you had a bigger budget for conducting surveys that year!) Would you expect your distribution of sample means to be wider (more variable) or narrower (more consistent)? Compare your resultant summary statistics to those predicted by the CLT.

A:

In [57]:
seed(47)
# calculate daily means from the larger sample size here
sample_mean1 = []
for i in range(365):
    daily_sample4 = townsfolk_sampler(50)
    sample_mean1.append(np.mean(daily_sample4))

_ = plt.hist(sample_mean1, bins=50)
_ = plt.xlabel('height (cm)')
_ = plt.ylabel('number of people')
_ = plt.title('Distribution of heights in sample size 10')

In [58]:
np.std(sample_mean1)
Out[58]:
0.6736107539771146
In [59]:
np.mean(sample_mean1)
Out[59]:
171.94366080916114
In [61]:
clt1 = 5/np.sqrt(50)
clt1
Out[61]:
0.7071067811865475
What we've seen so far, then, is that we can estimate population parameters from a sample from the population, and that samples have their own distributions. Furthermore, the larger the sample size, the narrower are those sampling distributions.

III Normally testing times!
All of the above is well and good. We've been sampling from a population we know is normally distributed, we've come to understand when to use $n$ and when to use $n-1$ in the denominator to calculate the spread of a distribution, and we've seen the Central Limit Theorem in action for a sampling distribution. All seems very well behaved in Frequentist land. But, well, why should we really care?

Remember, we rarely (if ever) actually know our population parameters but you still have to estimate them somehow. If we want to make inferences such as "is this observation unusual?" or "has my population mean changed?" then you need to have some idea of what the underlying distribution is so you can calculate relevant probabilities. In frequentist inference, you use the formulas above to deduce these population parameters. Take a moment in the next part of this assignment to refresh your understanding of how these probabilities work.

Recall some basic properties of the standard Normal distribution, such as about 68% of observations being within plus or minus 1 standard deviation of the mean.

Q: Using this fact, calculate the probability of observing the value 1 or less in a single observation from the standard normal distribution. Hint: you may find it helpful to sketch the standard normal distribution (the familiar bell shape) and mark the number of standard deviations from the mean on the x-axis and shade the regions of the curve that contain certain percentages of the population.

A:

In [62]:
normal_points = norm.rvs( 0, 1, 1000 )
In [65]:
probability =  sum( normal_points <=1 ) / 1000
probability
Out[65]:
0.844
Calculating this probability involved calculating the area under the pdf from the value of 1 and below. To put it another way, we need to integrate the pdf. We could just add together the known areas of chunks (from -Inf to 0 and then 0 to $+\sigma$ in the example above. One way to do this is using look up tables (literally). Fortunately, scipy has this functionality built in with the cdf() function.

Q: Use the cdf() function to answer the question above again and verify you get the same answer.

A:same value

In [67]:
x = norm.cdf(1)
print(x)
0.8413447460685429
Q: Using our knowledge of the population parameters for our townsfolk's heights, what is the probability of selecting one person at random and their height being 177 cm or less? Calculate this using both of the approaches given above.

A:

In [69]:
population = norm.rvs( 172, 5, 1000000 )
prob = sum(population <= 177)/ 1000000
prob
Out[69]:
0.840725
In [70]:
norm.cdf(177, 172, 5)
Out[70]:
0.8413447460685429
Q: Turning this question around. Let's say we randomly pick one person and measure their height and find they are 2.00 m tall? How surprised should we be at this result, given what we know about the population distribution? In other words, how likely would it be to obtain a value at least as extreme as this? Express this as a probability.

A:

In [ ]:

We could calculate this probability by virtue of knowing the population parameters. We were then able to use the known properties of the relevant normal distribution to calculate the probability of observing a value at least as extreme as our test value. We have essentially just performed a z-test (albeit without having prespecified a threshold for our "level of surprise")!

We're about to come to a pinch, though here. We've said a couple of times that we rarely, if ever, know the true population parameters; we have to estimate them from our sample and we cannot even begin to estimate the standard deviation from a single observation. This is very true and usually we have sample sizes larger than one. This means we can calculate the mean of the sample as our best estimate of the population mean and the standard deviation as our best estimate of the population standard deviation. In other words, we are now coming to deal with the sampling distributions we mentioned above as we are generally concerned with the properties of the sample means we obtain.

Above, we highlighted one result from the CLT, whereby the sampling distribution (of the mean) becomes narrower and narrower with the square root of the sample size. We remind ourselves that another result from the CLT is that even if the underlying population distribution is not normal, the sampling distribution will tend to become normal with sufficiently large sample size. This is the key driver for us 'requiring' a certain sample size, for example you may frequently see a minimum sample size of 30 stated in many places. In reality this is simply a rule of thumb; if the underlying distribution is approximately normal then your sampling distribution will already be pretty normal, but if the underlying distribution is heavily skewed then you'd want to increase your sample size.

Q: Let's now start from the position of knowing nothing about the heights of people in our town.

Use our favorite random seed of 47, to randomly sample the heights of 50 townsfolk
Estimate the population mean using np.mean
Estimate the population standard deviation using np.std (remember which denominator to use!)
Calculate the (95%) margin of error (use the exact critial z value to 2 decimal places - look this up or use norm.ppf())
Calculate the 95% Confidence Interval of the mean
Does this interval include the true population mean?
A:

In [71]:
seed(47)
# take your sample now
sample50 = townsfolk_sampler(50)
In [73]:
mu = np.mean(sample50)
mu
Out[73]:
172.7815108576788
In [74]:
sigma = np.std(sample50, ddof=1)
sigma
Out[74]:
4.195424364433547
In [76]:
margin_of_error = norm.ppf(0.975) * sigma
margin_of_error
Out[76]:
8.222880654151599
In [77]:
confidence_interval = (mu - margin_of_error, mu + margin_of_error)
confidence_interval
Out[77]:
(164.5586302035272, 181.0043915118304)
Q: Above we calculated the confidence interval using the critical z value. What is the problem with this? What requirement, or requirements, are we (strictly) failing?

A:

Q: Calculate the 95% confidence interval for the mean using the t distribution. Is this wider or narrower than that based on the normal distribution above? If you're unsure, you may find this resource useful. For calculating the critical value, remember how you could calculate this for the normal distribution using norm.ppf().

A:

In [78]:
t_margin_of_error = t.ppf(0.975 , 49) * sigma
In [80]:
t_confidence_interval = (mu - t_margin_of_error, mu + t_margin_of_error)
In [81]:
t_confidence_interval
Out[81]:
(164.35048995674052, 181.21253175861708)
This is slightly wider than the previous confidence interval. This reflects the greater uncertainty given that we are estimating population parameters from a sample.

Learning outcomes
Having completed this project notebook, you now have hands-on experience:

sampling and calculating probabilities from a normal distribution
the correct way to estimate the standard deviation of a population (the population parameter) from a sample
what a sampling distribution is and how the Central Limit Theorem applies
how to calculate critical values and confidence intervals
