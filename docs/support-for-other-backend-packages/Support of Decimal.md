---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

Should apply on the decimal value, with same dimension :
 - d.adjusted()
- d.conjugate() : Just returns self, this method is only to comply with the Decimal Specification.
- copy_abs() : Return the absolute value of the argument. This operation is unaffected by the context and is quiet: no flags are changed and no rounding is performed.
- copy_negate() : Return the negation of the argument. This operation is unaffected by the context and is quiet: no flags are changed and no rounding is performed.
- copy_sign(other, context=None) : Return a copy of the first operand with the sign set to be the same as the sign of the second operand.
- quantize(exp, rounding=None, context=None) : Return a value equal to the first operand after rounding and having the exponent of the second operand.
- rotate(other, context=None) : Return the result of rotating the digits of the first operand by an amount specified by the second operand.
- shift(other, context=None)
 
to_integral(rounding=None, context=None)
Identical to the to_integral_value() method. The to_integral name has been kept for compatibility with older versions.
 
to_integral_exact(rounding=None, context=None)
Round to the nearest integer, signaling Inexact or Rounded as appropriate if rounding occurs. The rounding mode is determined by the rounding parameter if given, else by the given context. If neither parameter is given then the rounding mode of the current context is used.
 
to_integral_value(rounding=None, context=None)
Round to the nearest integer without signaling Inexact or Rounded. If given, applies rounding; otherwise, uses the rounding method in either the supplied context or the current context.
 
Should apply on the decimal value, returns not a quantity :
 - is_canonical() : return True if the argument is canonical and False otherwise. Currently, a Decimal instance is always canonical, so this operation always returns True.
- is_finite() : Return True if the argument is a finite number, and False if the argument is an infinity or a NaN.
- is_infinite() : Return True if the argument is either positive or negative infinity and False otherwise.
- is_nan() : Return True if the argument is a (quiet or signaling) NaN and False otherwise.
- is_normal(context=None) : Return True if the argument is a normal finite number. Return False if the argument is zero, subnormal, infinite or a NaN.
- is_qnan() : Return True if the argument is a quiet NaN, and False otherwise.
- is_signed() : Return True if the argument has a negative sign and False otherwise. Note that zeros and NaNs can both carry signs.
- is_snan() : Return True if the argument is a signaling NaN and False otherwise.
- is_subnormal(context=None)  : Return True if the argument is subnormal, and False otherwise.
- is_zero() : Return True if the argument is a (positive or negative) zero and False otherwise.
 
Should be dimensionless, and return dimensionless :
 - d.exp(context=None)
- ln(context=None) : Return the natural (base e) logarithm of the operand.
- log10(context=None) : Return the base ten logarithm of the operand. The result is correctly rounded using the ROUND_HALF_EVEN rounding mode.
- logb(context=None) : For a nonzero number, return the adjusted exponent of its operand as a Decimal instance
 
Specific checks to do :
 - fma(other, third, context=None) : Fused multiply-add. Return self*other+third with no rounding of the intermediate product self*other.
- remainder_near(other, context=None) : Return the remainder from dividing self by other. This differs from self % other in that the sign of the remainder is chosen so as to minimize its absolute value
- scaleb(other, context=None) : Return the first operand with exponent adjusted by the second. Equivalently, return the first operand multiplied by 10**other. The second operand must be an integer.
- sqrt(context=None) : Return the square root of the argument to full precision.
- to_eng_string(context=None) :
 
SAme dimension binary :
 - max(other, context=None) : Like max(self, other) except that the context rounding rule is applied before returning and that NaN values are either signaled or ignored
- max_mag(other, context=None) : Similar to the max() method, but the comparison is done using the absolute values of the operands.
- min(other, context=None) : Like min(self, other) except that the context rounding rule is applied before returning and that NaN values are either signaled or ignored (depending on the context and whether they are signaling or quiet).
- min_mag(other, context=None) : Similar to the min() method, but the comparison is done using the absolute values of the operands.
- next_minus(context=None) : Return the largest number representable in the given context (or in the current thread’s context if no context is given) that is smaller than the given operand.
- next_plus(context=None) : Return the smallest number representable in the given context (or in the current thread’s context if no context is given) that is larger than the given operand.
- next_toward(other, context=None) : If the two operands are unequal, return the number closest to the first operand in the direction of the second operand. If both operands are numerically equal, return a copy of the first operand with the sign set to be the same as the sign of the second operand.
 
REturn not a quantity :
 - compare(other, context=None), compare_signal(other, context=None) : return a "Decimal boolean" : Compare the values of two Decimal instances. compare() returns a Decimal instance, and if either operand is a NaN then the result is a NaN:
 
To sort :
 - d.as_integer_ratio() : return a tuple of ints such that t[0]/t[1] = dec
- d.as_tuple() : Return a named tuple representation of the number: DecimalTuple(sign, digits, exponent).
- logical_and(other, context=None) : logical_and() is a logical operation which takes two logical operands (see Logical operands). The result is the digit-wise and of the two operands.
- logical_invert(context=None) : logical_invert() is a logical operation. The result is the digit-wise inversion of the operand.
- logical_or(other, context=None) : logical_or() is a logical operation which takes two logical operands (see Logical operands). The result is the digit-wise or of the two operands.
- logical_xor(other, context=None) :logical_xor() is a logical operation which takes two logical operands (see Logical operands). The result is the digit-wise exclusive or of the two operands.
- normalize(context=None) : Normalize the number by stripping the rightmost trailing zeros and converting any result equal to Decimal('0') to Decimal('0e0').
- radix() : Return Decimal(10), the radix (base) in which the Decimal class does all its arithmetic.
- same_quantum(other, context=None) : Test whether self and other have the same exponent or whether both are NaN
Class method :
 - Decimal.from_float(f)
 

```python
import decimal
from decimal import Decimal

from physipy import m
    
ctx = decimal.getcontext()
print(ctx.prec)
print(ctx.rounding)
 
Decimal(value='0', context=None)
x = Decimal((0, (3, 1, 4), -2))
 
decimal.Decimal(4).copy_negate()
 
data = list(map(Decimal, '1.34 1.87 3.45 2.35 1.00 0.03 9.25'.split()))
data = [d*m for d in data]
print(data)
max(data)
min(data)
sorted(data)
# sum(data) : can't work for 0 + m reason
a,b,c = data[:3]
str(a)
#float(a)
round(a, 1)
#int(a)
print(a * 5)
print(a * b)
print(c % a)
 
# drops the unit
(Decimal(2)*m).sqrt()
```

```python

```
