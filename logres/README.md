### DESCRIPTION

This is a logistic regression classifier. Again, two components of data points are drawn from gaussian distribution.

Surface of error function are shown below, which apparently a concave function.

![surface](https://raw.github.com/Oneplus/anothermlkid/master/logres/image/error_function_surface_without_regularization.png)

A newton-raphson method is played to approach the minimum.

The illustration of classification boundary here.

![boundary](https://raw.github.com/Oneplus/anothermlkid/master/logres/image/demo_without_regularization.png)

There seems no significant difference error function with penalty and without. But when the `C` value is set larger, iteration before converage reduce.
