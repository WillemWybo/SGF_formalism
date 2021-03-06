: This mod file is automaticaly generated by the ionc.write_mode_file() function in /source/ionchannels.py 

NEURON {
    SUFFIX INa
    USEION na WRITE ina
    RANGE  g, e
    GLOBAL var0inf, var1inf, tau0, tau1
    THREADSAFE
}

PARAMETER {
    g = 0.12 (S/cm2)
    e = 50.0 (mV)
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (mM) = (milli/liter)
}

ASSIGNED {
    ina (mA/cm2)
    var0inf
    tau0 (ms)
    var1inf
    tau1 (ms)
    v (mV)
}

STATE {
    var0
    var1
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = g * ( var0 * var0 * var0 * var1 *1.0) * (v - e)
}

INITIAL {
    rates(v)
    var0 = var0inf
    var1 = var1inf
}

DERIVATIVE states {
    rates(v)
    var0' = (var0inf - var0) / tau0
    var1' = (var1inf - var1) / tau1
}

PROCEDURE rates(v) {
    var0inf = (-0.1*v - 4.0)/(((-0.1*v - 4.0)/(exp(-0.1*v - 4.0) - 1.0) + 4.0*exp(-0.0555555555555556*v - 3.61111111111111))*(exp(-0.1*v - 4.0) - 1.0))
    tau0 = 1.0/(1.0*(-0.1*v - 4.0)/(exp(-0.1*v - 4.0) - 1.0) + 4.0*exp(-0.0555555555555556*v - 3.61111111111111))
    var1inf = 0.07*exp(-0.05*v - 3.25)/(0.07*exp(-0.05*v - 3.25) + 1.0/(exp(-0.1*v - 3.5) + 1.0))
    tau1 = 1.0/(0.07*exp(-0.05*v - 3.25) + 1.0/(exp(-0.1*v - 3.5) + 1.0))
}

