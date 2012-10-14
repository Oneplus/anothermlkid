#### DESCRIPTION
A parzen window demo for estimating probability density.

10,000 sample is randomly drawn from a 1D Gaussian distribution with mean = 5.0 and variance = 3.0.The histogram is shown below.

![hist](https://raw.github.com/Oneplus/anothermlkid/master/parzen/image/parzen_1d_demo_hist.png)

Two kinds of windows function __Hypercube__ and __Gaussian__ is implement, and different experiments is conducted. Results are shown below:

function=Hypercube, Vn=1.0/sqrt(N)

![function=Hypercube, Vn=1.0/sqrt(N)](https://raw.github.com/Oneplus/anothermlkid/master/parzen/image/parzen_1d_demo_1.png)

function=Gaussian, Vn=1.0/sqrt(N)

![function=Gaussian, Vn=1.0/sqrt(N)](https://raw.github.com/Oneplus/anothermlkid/master/parzen/image/parzen_1d_demo_2.png)

function=Hypercube, Vn=1.0/ln(N)

![function=Hypercube, Vn=1.0/ln(N)](https://raw.github.com/Oneplus/anothermlkid/master/parzen/image/parzen_1d_demo_3.png)

function=Gaussian, Vn=1.0/sqrt(N)

![function=Gaussian, Vn=1.0/sqrt(N)](https://raw.github.com/Oneplus/anothermlkid/master/parzen/image/parzen_1d_demo_4.png)

Apparently, smaller Vn make the graphs above very sharp. and __hypercube__ is sharper than __gaussian__ function.
