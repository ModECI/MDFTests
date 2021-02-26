"""
Wrap commonly-used torch builtins in nn.Module subclass
for easier automatic construction of script
"""
import torch.nn

class argmax(torch.nn.Module):
    def __init__(self):
        super(argmax, self).__init__()
    def forward(self, x):
        return torch.argmax(x)


# fns = [
# 'zeros','zeros_like','ones','ones_like','eye','empty','cat','chunk',
# 'dstack','gather','hstack','index_select','masked_select','movedim','narrow','nonzero',
# 'reshape','split','squeeze','stack','take','transpose','unbind','unsqueeze',
# 'vstack','where','bernoulli','multinomial','normal','poisson','rand','rand_like',
# 'randint','randint_like','randn','randn_like','randperm','absolute','acos','arccos',
# 'acosh','arccosh','add','addcdiv','addcmul','angle','asin','arcsin',
# 'asinh','arcsinh','atan','arctan','atanh','arctanh','atan2','bitwise_not',
# 'bitwise_and','bitwise_or','bitwise_xor','ceil','clamp','clip','conj','cos',
# 'cosh','deg2rad','div','divide','digamma','erf','erfc','erfinv',
# 'exp','exp2','expm1','fix','floor','floor_divide','fmod','frac',
# 'imag','lerp','lgamma','log','log10','log1p','log2','logaddexp',
# 'logaddexp2','logical_and','logical_not','logical_or','logical_xor','logit','hypot','i0',
# 'mul','multiply','mvlgamma','neg','negative','nextafter','polygamma','rad2deg',
# 'real','reciprocal','remainder','round','rsqrt','sigmoid','sign','signbit',
# 'sin','sinh','sqrt','square','sub','subtract','tan','tanh',
# 'true_divide','trunc','argmax','argmin','amax','amin','max','min',
# 'dist','logsumexp','mean','median','mode','norm','nansum','prod',
# 'quantile','nanquantile','std','std_mean','sum','unique','unique_consecutive','var',
# 'var_mean','count_nonzero','allclose','argsort','eq','equal','ge','greater_equal',
# 'gt','greater','isclose','isfinite','isinf','isposinf','isneginf','isnan',
# 'isreal','kthvalue','le','less_equal','lt','less','maximum','minimum','not_equal','sort','topk']


__all__ = ["argmax"]
